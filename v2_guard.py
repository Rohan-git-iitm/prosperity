import json
from typing import Any, List

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ═══════════════════════════════════════════════════════════════════
#  Logger — required by the IMC Prosperity 4 visualizer
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  IMC Prosperity 4 – Round 1 Trader  (v5)
#
#  Root cause of -100% position: passive skew was symmetric, so
#  bots kept hitting our ask when short AND our aggressive tier-2
#  sells kept firing, creating a one-way ratchet to -80.
#
#  v5 fix: ASYMMETRIC quadratic skew
#  - When SHORT: ask pushed UP to FV+11~16 (bots can't hit it)
#                bid stays at FV-1 (competitive, attracts rebalancing)
#  - When LONG:  bid pulled DOWN to FV-11~16 (bots can't hit it)
#                ask stays at FV+1 (competitive, attracts rebalancing)
#
#  Directional filter retained for tier-2 aggressive sells/buys:
#  - pos <= -60: suppress ALL sells (tier 2 + passive)
#  - pos >= +60: suppress ALL buys  (tier 1 + passive)
# ═══════════════════════════════════════════════════════════════════

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR         = 10_000
    ACO_SELL_LIMIT   = -60    # suppress ALL sells at or below this position
    ACO_BUY_LIMIT    =  60    # suppress ALL buys  at or above this position
    ACO_SKEW_LINEAR  =  6     # linear component of quadratic skew
    ACO_SKEW_QUAD    =  9     # quadratic amplification at extremes
    IPR_MAX_PREM     = 11

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0
        trader_data = ""

        for product in state.order_depths:
            position = state.position.get(product, 0)
            limit    = self.POSITION_LIMITS.get(product, 20)
            od       = state.order_depths[product]

            if product == "ASH_COATED_OSMIUM":
                orders = self._aco(product, od, position, limit)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self._ipr(product, od, position, limit)
            else:
                orders = []

            result[product] = orders

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    # ── ASH_COATED_OSMIUM — Asymmetric Market Making ────────────
    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders   = []
        fv       = self.ACO_FAIR
        buy_cap  = limit - position
        sell_cap = limit + position

        # Directional filter: stop ALL selling when too short,
        # stop ALL buying when too long (covers tier 2 aggressive takes)
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

        # ── Tier 3: buy ask == FV when short (at-FV rebalance) ──
        if can_buy:
            for ask, ask_vol in sorted(od.sell_orders.items()):
                if ask != fv or position >= 0 or buy_cap <= 0:
                    break
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap  -= qty
                    position += qty

        # ── Tier 4: sell bid == FV when long (at-FV rebalance) ──
        if can_sell:
            for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if bid != fv or position <= 0 or sell_cap <= 0:
                    break
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
                    position -= qty

        # ── Tier 5: asymmetric passive quotes ────────────────────
        r = position / limit   # position ratio in [-1, +1]

        # Ask: pushed UP when short, stays at FV+1 when long
        ask_push = max(0.0, -r)
        ask_off  = round(ask_push * self.ACO_SKEW_LINEAR
                         + ask_push ** 2 * self.ACO_SKEW_QUAD)
        ask_price = fv + 1 + ask_off   # always >= FV+1

        # Bid: pulled DOWN when long, stays at FV-1 when short
        bid_pull = max(0.0, r)
        bid_off  = round(bid_pull * self.ACO_SKEW_LINEAR
                         + bid_pull ** 2 * self.ACO_SKEW_QUAD)
        bid_price = fv - 1 - bid_off   # always <= FV-1

        if buy_cap > 0 and can_buy:
            orders.append(Order(product, bid_price, buy_cap))
        if sell_cap > 0 and can_sell:
            orders.append(Order(product, ask_price, -sell_cap))

        return orders

    # ── INTARIAN_PEPPER_ROOT — Trend Following (max long) ───────
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
                best_ask    = min(od.sell_orders)
                passive_bid = best_ask - 1
                if fair_value is not None:
                    passive_bid = min(passive_bid, int(fair_value) + 8)
            elif od.buy_orders:
                passive_bid = max(od.buy_orders) + 1
            else:
                passive_bid = 13_000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders