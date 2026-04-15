import json
from typing import Any, List

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ═══════════════════════════════════════════════════════════════════
#  Logger — required by the IMC Prosperity 4 visualizer
#  Use logger.print() instead of print() everywhere.
#  logger.flush() is called at the end of Trader.run().
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
#  IMC Prosperity 4 – Round 1 Trader  (v2)
#
#  [1] ACO: Exploit at-fair-value standing orders
#  [2] ACO: Tighter passive spread FV±1
#  [3] IPR: Real-time fair value + buy cap at FV+11
# ═══════════════════════════════════════════════════════════════════

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR       = 10_000
    ACO_MM_EDGE    = 1
    ACO_SKEW_TICKS = 2
    IPR_MAX_PREM   = 11

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

    # ── ASH_COATED_OSMIUM — 4-tier Market Making ────────────────
    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders   = []
        fv       = self.ACO_FAIR
        buy_cap  = limit - position
        sell_cap = limit + position

        # Tier 1 & 3: buy-side takes
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask < fv:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty
            elif ask == fv and position < 0:
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap  -= qty
                    position += qty

        # Tier 2 & 4: sell-side takes
        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if sell_cap <= 0:
                break
            if bid > fv:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty
            elif bid == fv and position > 0:
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
                    position -= qty

        # Tier 5: passive market-making
        best_bid = max(od.buy_orders)  if od.buy_orders  else fv - 8
        best_ask = min(od.sell_orders) if od.sell_orders else fv + 8

        raw_bid = min(best_bid + 1, fv - self.ACO_MM_EDGE)
        raw_ask = max(best_ask - 1, fv + self.ACO_MM_EDGE)

        skew_ticks = round((position / limit) * self.ACO_SKEW_TICKS)
        bid_price  = raw_bid  - skew_ticks
        ask_price  = raw_ask  - skew_ticks

        if bid_price >= ask_price:
            bid_price = fv - 1
            ask_price = fv + 1

        if buy_cap > 0:
            orders.append(Order(product, bid_price,  buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -sell_cap))

        return orders

    # ── INTARIAN_PEPPER_ROOT — Trend Following (max long) ───────
    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders  = []
        buy_cap = limit - position

        if buy_cap <= 0:
            return orders

        # Real-time fair value from order book mid
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

        # Hit asks up to price cap
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask > max_buy:
                break
            qty = min(-ask_vol, buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap -= qty

        # Passive bid for remaining capacity
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