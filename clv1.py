import json
from typing import Any, List, Dict, Tuple, Optional

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ═══════════════════════════════════════════════════════════════════════════
#  Logger — required for IMC Prosperity 4 visualizer
# ═══════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [
            [t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
            for arr in trades.values() for t in arr
        ]

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice,
                observation.transportFees, observation.exportTariff,
                observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ═══════════════════════════════════════════════════════════════════════════
#  Trader
# ═══════════════════════════════════════════════════════════════════════════
class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ── ACO hyperparameters ─────────────────────────────────────────────────
    ACO_FAIR = 10_000          # hard-coded mean-reversion anchor (from ADF analysis)

    # Passive ladder: fraction of remaining capacity at each price level
    # 50% at front, 30% one tick back, 20% two ticks back
    ACO_LADDER_FRACS = (0.50, 0.30, 0.20)

    # Reservation price skew: how many ticks to shift the whole ladder per unit
    # of inventory utilisation (0→1). Higher = more aggressive rebalancing.
    ACO_SKEW_TICKS = 3
    ACO_SHAPE_K    = 1.2       # >1 → skew accelerates as position approaches limit

    # Minimum edge: ladder front is always at least this many ticks from fair value
    ACO_BASE_EDGE  = 1

    # Per-step ladder cap: max units submitted per side per timestep.
    # Prevents the visualizer ±100% artefact and limits adverse-selection exposure.
    ACO_MAX_STEP = 20

    # Mean-reversion recovery (triggered when |position| > ACO_RECOVERY_START)
    # Only fires when mid is within MR_THRESHOLD ticks of fair value (timing gate).
    ACO_RECOVERY_START = 50
    ACO_MR_THRESHOLD   = 3     # ticks from FV before we consider mid "safe to trade"
    ACO_RECOVERY_QTY   = 15    # max units per recovery order

    # Tiered premium bids/asks for recovery (position threshold → extra ticks)
    # Short recovery: we bid ABOVE fair value to attract sellers
    ACO_SHORT_RECOVERY_TIERS = [
        (-50, 0),   # pos ≤ -50: bid @ FV+0
        (-60, 2),   # pos ≤ -60: bid @ FV+2
        (-70, 4),   # pos ≤ -70: bid @ FV+4
        (-75, 6),   # pos ≤ -75: bid @ FV+6 (urgent)
    ]
    # Long recovery: we ask BELOW fair value to attract buyers
    ACO_LONG_RECOVERY_TIERS = [
        (50,  0),   # pos ≥ +50: ask @ FV-0
        (60,  2),   # pos ≥ +60: ask @ FV-2
        (70,  4),   # pos ≥ +70: ask @ FV-4
        (75,  6),   # pos ≥ +75: ask @ FV-6 (urgent)
    ]

    # ── IPR hyperparameters ─────────────────────────────────────────────────
    IPR_MAX_PREM = 11          # don't chase asks more than mid + this many ticks

    # ───────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Load persistent state (mid history for ACO micro-price smoothing)
        if state.traderData:
            try:
                memory = json.loads(state.traderData)
            except json.JSONDecodeError:
                memory = {}
        else:
            memory = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            limit    = self.POSITION_LIMITS.get(product, 20)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self._aco(product, od, position, limit)

            elif product == "INTARIAN_PEPPER_ROOT":
                result[product] = self._ipr(product, od, position, limit)

        trader_data_out = json.dumps(memory)
        logger.flush(state, result, conversions, trader_data_out)
        return result, conversions, trader_data_out

    # ═══════════════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM
    #
    #  Strategy:
    #    Tiers 1–4  — risk-free aggressive takes (mispriced orders + FV rebalance)
    #    Tier  5    — mean-reversion-timed position recovery with tiered premiums
    #    Tier  6    — passive ladder MM anchored to micro-price with reservation skew
    # ═══════════════════════════════════════════════════════════════════════
    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders: List[Order] = []

        # ── Micro-price fair value (from sv45) ──────────────────────────────
        # Volume-weighted mid: if big ask volume → price should trade lower, so
        # micro-price < simple mid. More accurate than plain (bid+ask)/2.
        best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is not None and best_ask is not None:
            vol_bid = od.buy_orders[best_bid]
            vol_ask = abs(od.sell_orders[best_ask])
            micro   = (best_bid * vol_ask + best_ask * vol_bid) / (vol_bid + vol_ask)
            fv      = micro
        else:
            fv = float(self.ACO_FAIR)

        int_fv = int(round(fv))

        # Hard anchor for aggressive takes and rebalancing — always ACO_FAIR (10000).
        # We can't use micro_price here because a mispriced order IN the book will
        # pull the micro-price toward itself, making int_fv too low/high and causing
        # us to miss the very trade we're trying to take.
        take_fv = self.ACO_FAIR

        # ── Tier 1: Take asks strictly below fair value (free profit) ───────
        for ask in sorted(od.sell_orders.keys()):
            if ask >= take_fv:
                break
            if position >= limit:
                break
            avail = -od.sell_orders[ask]
            qty   = min(avail, limit - position)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                position += qty

        # ── Tier 2: Take bids strictly above fair value (free profit) ────────
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid <= take_fv:
                break
            if position <= -limit:
                break
            avail = od.buy_orders[bid]
            qty   = min(avail, limit + position)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                position -= qty

        # ── Tier 3: Buy from ask == FV to flatten short (free rebalance) ────
        if position < 0 and take_fv in od.sell_orders:
            avail = -od.sell_orders[take_fv]
            qty   = min(avail, -position, limit - position)
            if qty > 0:
                orders.append(Order(product, take_fv, qty))
                position += qty

        # ── Tier 4: Sell into bid == FV to flatten long (free rebalance) ────
        if position > 0 and take_fv in od.buy_orders:
            avail = od.buy_orders[take_fv]
            qty   = min(avail, position, limit + position)
            if qty > 0:
                orders.append(Order(product, take_fv, -qty))
                position -= qty

        # ── Tier 5: Mean-reversion-timed recovery ───────────────────────────
        # Use ACO_FAIR (10000) as the stable anchor for timing gate — mid is
        # compared to the known fundamental value, not the current micro-price.
        mid     = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else float(self.ACO_FAIR)
        anchor  = self.ACO_FAIR

        if position < -self.ACO_RECOVERY_START:
            rec_price, rec_qty = self._short_recovery(position, mid, anchor, limit)
            if rec_price is not None and rec_qty > 0:
                orders.append(Order(product, rec_price, rec_qty))
                position += rec_qty

        elif position > self.ACO_RECOVERY_START:
            rec_price, rec_qty = self._long_recovery(position, mid, anchor, limit)
            if rec_price is not None and rec_qty > 0:
                orders.append(Order(product, rec_price, -rec_qty))
                position -= rec_qty

        # ── Tier 6: Passive ladder MM (from sv45) ───────────────────────────
        # 1. Compute reservation price — shifts the entire ladder based on
        #    inventory. If long, reservation < FV (want to sell cheaper).
        #    If short, reservation > FV (want to buy more expensively).
        utilization      = position / limit              # −1 to +1
        penalty          = (abs(utilization) ** self.ACO_SHAPE_K) * self.ACO_SKEW_TICKS
        inventory_offset = penalty if position > 0 else -penalty
        reservation      = fv - inventory_offset

        # 2. "Penny the market": front of ladder is one tick inside best bid/ask,
        #    but bounded by (reservation ± ACO_BASE_EDGE) to protect against
        #    posting on the wrong side of fair value.
        liq_bid = (max(od.buy_orders.keys())  if od.buy_orders  else int_fv - 8)
        liq_ask = (min(od.sell_orders.keys()) if od.sell_orders else int_fv + 8)

        front_bid = min(liq_bid + 1, int(reservation) - self.ACO_BASE_EDGE)
        front_ask = max(liq_ask - 1, int(reservation) + self.ACO_BASE_EDGE)

        # Safety: never let passive orders cross existing best bid/ask
        if best_ask is not None:
            front_bid = min(front_bid, best_ask - 1)
        if best_bid is not None:
            front_ask = max(front_ask, best_bid + 1)

        # Safety fallback: if bid/ask inverted, pin to reservation ± 1
        if front_bid >= front_ask:
            front_bid = int(reservation) - 1
            front_ask = int(reservation) + 1

        # 3. Capacity available after tiers 1–5, capped by ACO_MAX_STEP
        buy_cap  = min(limit - position, self.ACO_MAX_STEP)
        sell_cap = min(limit + position, self.ACO_MAX_STEP)

        # 4. Post 3-level ladder: 50% front, 30% one tick back, 20% two ticks back
        #    Splitting into smaller orders gets better fill rates — bots with limited
        #    inventory fill the front level, others fill deeper levels.
        if buy_cap > 0:
            q1 = max(1, int(buy_cap * self.ACO_LADDER_FRACS[0]))
            q2 = max(1, int(buy_cap * self.ACO_LADDER_FRACS[1]))
            q3 = max(0, buy_cap - q1 - q2)
            if q1 > 0: orders.append(Order(product, front_bid,     q1))
            if q2 > 0: orders.append(Order(product, front_bid - 1, q2))
            if q3 > 0: orders.append(Order(product, front_bid - 2, q3))

        if sell_cap > 0:
            q1 = max(1, int(sell_cap * self.ACO_LADDER_FRACS[0]))
            q2 = max(1, int(sell_cap * self.ACO_LADDER_FRACS[1]))
            q3 = max(0, sell_cap - q1 - q2)
            if q1 > 0: orders.append(Order(product, front_ask,     -q1))
            if q2 > 0: orders.append(Order(product, front_ask + 1, -q2))
            if q3 > 0: orders.append(Order(product, front_ask + 2, -q3))

        return orders

    def _short_recovery(self, position: int, mid: float,
                         fv: int, limit: int) -> Tuple[Optional[int], int]:
        """
        When short (position < -ACO_RECOVERY_START):
        Place a recovery bid ABOVE fair value to attract sellers.
        Timing gate: only fire when mid has reverted close to FV.
        """
        # Timing gate: if mid is still elevated, wait for mean reversion
        if mid > fv + self.ACO_MR_THRESHOLD:
            return None, 0

        # Find appropriate premium tier.
        # Sort most-extreme (most negative) first, break on first match so we
        # don't accidentally overwrite with a shallower tier.
        premium = 0
        for threshold, ticks in sorted(self.ACO_SHORT_RECOVERY_TIERS):  # e.g. -75,-70,-60,-50
            if position <= threshold:
                premium = ticks
                break

        recovery_price = fv + premium
        # Only recover enough to reach -ACO_RECOVERY_START, capped at ACO_RECOVERY_QTY
        target = min(-position - self.ACO_RECOVERY_START,
                     self.ACO_RECOVERY_QTY,
                     limit - position)
        return recovery_price, max(0, target)

    def _long_recovery(self, position: int, mid: float,
                        fv: int, limit: int) -> Tuple[Optional[int], int]:
        """
        When long (position > +ACO_RECOVERY_START):
        Place a recovery ask BELOW fair value to attract buyers.
        Timing gate: only fire when mid has reverted close to FV.
        """
        # Timing gate: if mid is depressed, wait for mean reversion
        if mid < fv - self.ACO_MR_THRESHOLD:
            return None, 0

        # Find appropriate discount tier (most extreme first, break on first match)
        discount = 0
        for threshold, ticks in sorted(self.ACO_LONG_RECOVERY_TIERS, reverse=True):  # 75,70,60,50
            if position >= threshold:
                discount = ticks
                break

        recovery_price = fv - discount
        target = min(position - self.ACO_RECOVERY_START,
                     self.ACO_RECOVERY_QTY,
                     limit + position)
        return recovery_price, max(0, target)

    # ═══════════════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT
    #
    #  Strategy: pure trend following. Price rises ~+1000/day (+0.1 per step).
    #  Buy to +80 as fast as possible. Never sell. No liquidation.
    #
    #  Why no liquidation?
    #  - IPR trends upward across BOTH live trading days
    #  - Selling at day 1 end and rebuying at day 2 start costs the overnight gain
    #    (day 2 opens ~1000 higher than day 1 close)
    #  - deepv9's liquidation was designed for a single-day tournament; this is 2-day
    # ═══════════════════════════════════════════════════════════════════════
    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders:  List[Order] = []
        buy_cap = limit - position

        if buy_cap <= 0:
            return orders   # already maxed out, nothing to do

        # Compute current mid
        if od.buy_orders and od.sell_orders:
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        elif od.sell_orders:
            mid = min(od.sell_orders) - 6.5
        elif od.buy_orders:
            mid = max(od.buy_orders) + 6.5
        else:
            mid = None

        max_price = (mid + self.IPR_MAX_PREM) if mid is not None else float("inf")

        # Hit every ask up to mid + IPR_MAX_PREM
        for ask in sorted(od.sell_orders.keys()):
            if buy_cap <= 0:
                break
            if ask > max_price:
                break
            qty = min(-od.sell_orders[ask], buy_cap)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                buy_cap -= qty

        # Place an aggressive passive bid just below the best ask to catch
        # any sells that come in at the next step
        if buy_cap > 0:
            if od.sell_orders:
                passive_bid = min(od.sell_orders) - 1
                if mid is not None:
                    passive_bid = min(passive_bid, int(mid) + 10)
            elif od.buy_orders:
                passive_bid = max(od.buy_orders) + 1
            else:
                passive_bid = 13_000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders
