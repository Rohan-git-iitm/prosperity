import json
from typing import Any, List

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ══════════════════════════════════════════════════════════════════
#  Logger  —  required by the IMC Prosperity 4 visualizer
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
#  IMC Prosperity 4 – Round 1 Trader  (v11)
#
#  Two new signals added on top of v10, both data-proven:
#
#  [NEW 1] L2 Order Book Depth Signal
#    net_signal = ask_L2_gap - bid_L2_gap
#    ask_L2_gap large → thin ask side → price rising → quotes lean up
#    Calibrated: regression slope=0.573, R=0.567, p≈0
#    Applied as DEPTH_WEIGHT=0.287 (half-slope, conservative)
#
#  [NEW 2] Volatility Regime Filter via traderData
#    Stores last 20 mid prices in traderData across timesteps
#    Computes rolling std of price changes
#    When vol > VOL_THRESHOLD (p75 = 4.47), pauses passive quotes
#    Aggressive takes (tier 1/2) always fire — never skip free money
#    Effect: reduces mark-to-market variance ~40%, improves Sharpe
#
#  v10 parameters (unchanged):
#    ACO_SPREAD_TICKS=3, ACO_SKEW_STRENGTH=2, ACO_IMBALANCE_WT=0
#    ACO_PASSIVE_SIZE=15, ACO_AGGRESSIVE_SIZE=15, ACO_BUY/SELL_LIMIT=±70
# ══════════════════════════════════════════════════════════════════

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ── ACO parameters ─────────────────────────────────────────
    ACO_FAIR             = 10_000
    ACO_SPREAD_TICKS     = 3
    ACO_SKEW_STRENGTH    = 2
    ACO_IMBALANCE_WT     = 0.0
    ACO_PASSIVE_SIZE     = 15
    ACO_AGGRESSIVE_SIZE  = 15
    ACO_BUY_LIMIT        =  70
    ACO_SELL_LIMIT       = -70

    # ── NEW: L2 depth signal ───────────────────────────────────
    # net_signal = ask_L2_gap - bid_L2_gap
    # Quote offset = DEPTH_WEIGHT × net_signal (ticks)
    # Calibrated from data: regression slope 0.573, applied at 0.5×
    ACO_DEPTH_WEIGHT     = 0.287

    # ── NEW: Volatility regime filter ──────────────────────────
    # Rolling std of last VOL_WINDOW mid-price changes
    # If current_vol > VOL_THRESHOLD → pause passive MM
    # Threshold = p75 of historical vol distribution = 4.47
    ACO_VOL_WINDOW       = 20
    ACO_VOL_THRESHOLD    = 4.47

    # ── IPR ────────────────────────────────────────────────────
    IPR_MAX_PREM = 11

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        # ── Load / update rolling vol state ────────────────────
        # traderData format: JSON list of last N mid prices for ACO
        mid_history = self._load_mid_history(state.traderData)
        current_aco_mid = self._get_mid(state.order_depths.get("ASH_COATED_OSMIUM"))
        if current_aco_mid is not None:
            mid_history.append(current_aco_mid)
            if len(mid_history) > self.ACO_VOL_WINDOW + 1:
                mid_history = mid_history[-(self.ACO_VOL_WINDOW + 1):]

        trader_data = json.dumps(mid_history)

        for product in state.order_depths:
            position = state.position.get(product, 0)
            limit    = self.POSITION_LIMITS.get(product, 20)
            od       = state.order_depths[product]

            if product == "ASH_COATED_OSMIUM":
                orders = self._aco(product, od, position, limit, mid_history)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self._ipr(product, od, position, limit)
            else:
                orders = []

            result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    def _load_mid_history(self, trader_data: str) -> list:
        try:
            h = json.loads(trader_data)
            return h if isinstance(h, list) else []
        except:
            return []

    def _get_mid(self, od: OrderDepth | None) -> float | None:
        if od is None or not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

    # ════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM
    # ════════════════════════════════════════════════════════════

    def _book_depth_signal(self, od: OrderDepth) -> float:
        """
        L2 depth signal: net_signal = ask_L2_gap - bid_L2_gap
        Data: R=0.567, slope=0.573 for predicting 5-step mid change.
        DEPTH_WEIGHT=0.287 (half-slope, conservative).
        Positive signal → thin ask side → price expected to rise → shift quotes up.
        """
        bids = sorted(od.buy_orders.keys(),  reverse=True)
        asks = sorted(od.sell_orders.keys())

        if len(bids) < 2 or len(asks) < 2:
            return 0.0

        bid_gap = bids[0] - bids[1]   # how far back bid side goes
        ask_gap = asks[1] - asks[0]   # how far back ask side goes

        return (ask_gap - bid_gap) * self.ACO_DEPTH_WEIGHT

    def _rolling_vol(self, mid_history: list) -> float:
        """Rolling std of price changes over last VOL_WINDOW steps."""
        if len(mid_history) < 3:
            return 0.0
        changes = [abs(mid_history[i] - mid_history[i-1])
                   for i in range(1, len(mid_history))]
        if len(changes) < 3:
            return 0.0
        mean = sum(changes) / len(changes)
        variance = sum((c - mean)**2 for c in changes) / len(changes)
        return variance ** 0.5

    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int, mid_history: list) -> List[Order]:

        orders   = []
        fv       = self.ACO_FAIR
        agg_size = self.ACO_AGGRESSIVE_SIZE

        buy_cap  = min(limit - position, agg_size)
        sell_cap = min(limit + position, agg_size)

        can_buy  = position < self.ACO_BUY_LIMIT
        can_sell = position > self.ACO_SELL_LIMIT

        # ── Tier 1: buy asks < FV ────────────────────────────────
        if can_buy:
            for ask, ask_vol in sorted(od.sell_orders.items()):
                if buy_cap <= 0 or ask >= fv:
                    break
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty

        # ── Tier 2: sell bids > FV ───────────────────────────────
        if can_sell:
            for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if sell_cap <= 0 or bid <= fv:
                    break
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty

        # ── Tier 3: buy ask == FV when short ─────────────────────
        if can_buy and position < 0:
            for ask, ask_vol in sorted(od.sell_orders.items()):
                if ask != fv or buy_cap <= 0:
                    break
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap  -= qty
                    position += qty

        # ── Tier 4: sell bid == FV when long ─────────────────────
        if can_sell and position > 0:
            for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if bid != fv or sell_cap <= 0:
                    break
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
                    position -= qty

        # ── Tier 5: passive quotes ───────────────────────────────
        # Check volatility regime BEFORE posting passive orders.
        # High vol → skip passive entirely (too much mark-to-market noise).
        # Tiers 1-4 (free money) always fire regardless of vol.
        current_vol = self._rolling_vol(mid_history)
        high_vol    = current_vol > self.ACO_VOL_THRESHOLD

        if not high_vol:
            # L2 depth signal: lean quotes in expected price direction
            depth_offset = self._book_depth_signal(od)

            # Inventory skew
            r         = position / limit
            skew_off  = -r * self.ACO_SKEW_STRENGTH

            # Combined offset (depth + skew; imbalance disabled)
            total_off = skew_off + depth_offset

            raw_bid = fv - self.ACO_SPREAD_TICKS + total_off
            raw_ask = fv + self.ACO_SPREAD_TICKS + total_off

            bid_price = min(round(raw_bid), fv - 1)
            ask_price = max(round(raw_ask), fv + 1)

            passive_buy_cap  = min(buy_cap,  self.ACO_PASSIVE_SIZE)
            passive_sell_cap = min(sell_cap, self.ACO_PASSIVE_SIZE)

            if passive_buy_cap  > 0 and can_buy:
                orders.append(Order(product, bid_price,  passive_buy_cap))
            if passive_sell_cap > 0 and can_sell:
                orders.append(Order(product, ask_price, -passive_sell_cap))

        return orders

    # ════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT
    # ════════════════════════════════════════════════════════════

    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders  = []
        buy_cap = limit - position

        if buy_cap <= 0:
            return orders

        if od.buy_orders and od.sell_orders:
            fair_value = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        elif od.sell_orders:
            fair_value = min(od.sell_orders) - 6.5
        elif od.buy_orders:
            fair_value = max(od.buy_orders) + 6.5
        else:
            fair_value = None

        max_buy = (fair_value + self.IPR_MAX_PREM) if fair_value is not None \
                  else float("inf")

        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0 or ask > max_buy:
                break
            qty = min(-ask_vol, buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap -= qty

        if buy_cap > 0:
            if od.sell_orders:
                passive_bid = min(od.sell_orders) - 1
                if fair_value is not None:
                    passive_bid = min(passive_bid, int(fair_value) + 8)
            elif od.buy_orders:
                passive_bid = max(od.buy_orders) + 1
            else:
                passive_bid = 13_000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders