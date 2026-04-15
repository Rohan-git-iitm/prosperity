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


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR = 10000
    ACO_SKEW_TICKS = 2
    IPR_MAX_PREM = 11

    def run(self, state: TradingState):
        result = {}
        
        trader_data = state.traderData
        if trader_data:
            try:
                state_data = json.loads(trader_data)
            except json.JSONDecodeError:
                state_data = {"ipr_ema": None}
        else:
            state_data = {"ipr_ema": None}

        for product in state.order_depths:
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)
            od = state.order_depths[product]

            if product == "ASH_COATED_OSMIUM":
                orders = self._aco(product, od, position, limit)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders, new_ema = self._ipr(product, od, position, limit, state_data.get("ipr_ema"))
                state_data["ipr_ema"] = new_ema
            else:
                orders = []

            result[product] = orders

        return result, 1, json.dumps(state_data)

    def _get_liquidity_best(self, orders_dict, is_buy, threshold=5):
        if not orders_dict:
            return None
        sorted_prices = sorted(orders_dict.keys(), reverse=is_buy)
        for price in sorted_prices:
            if abs(orders_dict[price]) >= threshold:
                return price
        return sorted_prices[0]

    def _calculate_obi(self, od):
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _aco(self, product: str, od: OrderDepth, position: int, limit: int) -> List[Order]:
        orders = []
        buy_cap = limit - position
        sell_cap = limit + position

        obi = self._calculate_obi(od)
        dynamic_fair = self.ACO_FAIR + (1 if obi > 0.5 else (-1 if obi < -0.5 else 0))

        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask < self.ACO_FAIR:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty
            elif ask == self.ACO_FAIR and position < 0:
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
                    position += qty

        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if sell_cap <= 0:
                break
            if bid > self.ACO_FAIR:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty
            elif bid == self.ACO_FAIR and position > 0:
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
                    position -= qty

        best_bid = self._get_liquidity_best(od.buy_orders, True, 5)
        if best_bid is None: 
            best_bid = dynamic_fair - 8
            
        best_ask = self._get_liquidity_best(od.sell_orders, False, 5)
        if best_ask is None: 
            best_ask = dynamic_fair + 8

        spread = best_ask - best_bid
        mm_edge = 2 if spread > 4 else 1

        raw_bid = min(best_bid + 1, dynamic_fair - mm_edge)
        raw_ask = max(best_ask - 1, dynamic_fair + mm_edge)

        skew_ticks = round((position / limit) * self.ACO_SKEW_TICKS)
        bid_price = int(raw_bid - skew_ticks)
        ask_price = int(raw_ask - skew_ticks)

        if bid_price >= ask_price:
            bid_price = int(dynamic_fair - 1)
            ask_price = int(dynamic_fair + 1)

        if buy_cap > 0:
            orders.append(Order(product, bid_price, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -sell_cap))

        return orders

    def _ipr(self, product: str, od: OrderDepth, position: int, limit: int, prev_ema: float):
        orders = []
        buy_cap = limit - position

        if od.buy_orders and od.sell_orders:
            current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        elif od.sell_orders:
            current_mid = min(od.sell_orders) - 6.5
        elif od.buy_orders:
            current_mid = max(od.buy_orders) + 6.5
        else:
            current_mid = None

        if current_mid is not None:
            alpha = 0.2
            new_ema = current_mid if prev_ema is None else (current_mid * alpha) + (prev_ema * (1 - alpha))
            fair_value = new_ema
        else:
            new_ema = prev_ema
            fair_value = prev_ema

        max_buy = (fair_value + self.IPR_MAX_PREM) if fair_value is not None else float("inf")

        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask > max_buy:
                break
            qty = min(-ask_vol, buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap -= qty

        if buy_cap > 0:
            if od.sell_orders:
                best_ask = min(od.sell_orders)
                passive_bid = best_ask - 1
                if fair_value is not None:
                    passive_bid = min(passive_bid, int(fair_value) + 8)
            elif od.buy_orders:
                passive_bid = max(od.buy_orders) + 1
            else:
                passive_bid = 13000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders, new_ema