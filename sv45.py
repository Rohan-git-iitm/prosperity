import json
from typing import Any, List
from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState
)

# ═══════════════════════════════════════════════════════════════════
#  Logger
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
        return [[getattr(l, 'symbol', l.get('symbol')), getattr(l, 'product', l.get('product')), getattr(l, 'denomination', l.get('denomination'))] if isinstance(l, dict) else [l.symbol, l.product, l.denomination] for l in listings.values()]

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
#  Trader
# ═══════════════════════════════════════════════════════════════════
class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR = 10000
    IPR_MAX_PREM = 11

    # ==========================================
    # HYPERPARAMETERS
    # ==========================================
    ACO_SKEW_TICKS = 2     
    ACO_SHAPE_K    = 1.0   
    ACO_BASE_EDGE  = 1
    
    # 3-Tier Ladder Fractions (User Optimized)
    ACO_LADDER_FRACS = (0.50, 0.30, 0.20)
    # ==========================================

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
            limit    = self.POSITION_LIMITS.get(product, 20)
            od       = state.order_depths[product]

            if product == "ASH_COATED_OSMIUM":
                orders = self._aco(product, od, position, limit)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders, new_ema = self._ipr(product, od, position, limit, state_data.get("ipr_ema"))
                state_data["ipr_ema"] = new_ema
            else:
                orders = []

            result[product] = orders

        trader_data_out = json.dumps(state_data)
        conversions = 0  
        
        # Uncomment this line if submitting to the official IMC servers!
        logger.flush(state, result, conversions, trader_data_out)
        
        return result, conversions, trader_data_out

    def _get_liquidity_best(self, orders_dict, is_buy, threshold=5):
        if not orders_dict:
            return None
        sorted_prices = sorted(orders_dict.keys(), reverse=is_buy)
        for price in sorted_prices:
            if abs(orders_dict[price]) >= threshold:
                return price
        return sorted_prices[0]

    def _aco(self, product: str, od: OrderDepth, position: int, limit: int) -> List[Order]:
        orders = []
        buy_cap  = limit - position
        sell_cap = limit + position

        # ---------------------------------------------------------
        # 1. MICRO-PRICE CALCULATION
        # ---------------------------------------------------------
        best_bid_1 = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask_1 = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid_1 and best_ask_1:
            vol_bid = od.buy_orders[best_bid_1]
            vol_ask = abs(od.sell_orders[best_ask_1])
            micro_price = (best_bid_1 * vol_ask + best_ask_1 * vol_bid) / (vol_bid + vol_ask)
            fv = micro_price
        else:
            fv = float(self.ACO_FAIR)

        int_fv = int(fv) # Used for aggressive tiers to prevent float matching errors

        # ---------------------------------------------------------
        # 2. AGGRESSIVE TIERS (1-4)
        # ---------------------------------------------------------
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0: break
            if ask < int_fv:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty  

        # Tier 2: Buy at fv to flatten a short position
        if int_fv in od.sell_orders and position < 0 and buy_cap > 0:
            ask_vol = od.sell_orders[int_fv]
            qty = min(-ask_vol, buy_cap, -position)
            if qty > 0:
                orders.append(Order(product, int_fv, qty))
                buy_cap  -= qty
                position += qty

        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if sell_cap <= 0: break
            if bid > int_fv:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty

        # Tier 4: Sell at fv to flatten a long position
        if int_fv in od.buy_orders and position > 0 and sell_cap > 0:
            bid_vol = od.buy_orders[int_fv]
            qty = min(bid_vol, sell_cap, position)
            if qty > 0:
                orders.append(Order(product, int_fv, -qty))
                sell_cap -= qty
                position -= qty

        # ---------------------------------------------------------
        # 3. TIER 5: MARKET-ANCHORED LADDER
        # ---------------------------------------------------------
        best_bid_liq = self._get_liquidity_best(od.buy_orders, True, 5)
        if best_bid_liq is None: best_bid_liq = int_fv - 8

        best_ask_liq = self._get_liquidity_best(od.sell_orders, False, 5)
        if best_ask_liq is None: best_ask_liq = int_fv + 8

        # Calculate Reservation Price to protect against max inventory
        utilization = abs(position) / limit
        penalty_magnitude = (utilization ** self.ACO_SHAPE_K) * self.ACO_SKEW_TICKS
        inventory_offset = penalty_magnitude if position > 0 else -penalty_magnitude
        reservation_price = fv - inventory_offset

        # The Front Line: Penny the market, strictly bounded by Reservation Price
        front_bid = min(best_bid_liq + 1, int(reservation_price) - self.ACO_BASE_EDGE)
        front_ask = max(best_ask_liq - 1, int(reservation_price) + self.ACO_BASE_EDGE)

        # Strict safety to ensure we NEVER accidentally submit a passive order that crosses the spread
        if best_ask_1 is not None:
            front_bid = min(front_bid, best_ask_1 - 1)
        if best_bid_1 is not None:
            front_ask = max(front_ask, best_bid_1 + 1)

        # Ultimate safety fallback
        if front_bid >= front_ask:
            front_bid = int(reservation_price) - 1
            front_ask = int(reservation_price) + 1

        # Bids (Buying) Ladder
        if buy_cap > 0:
            q1 = int(buy_cap * self.ACO_LADDER_FRACS[0])
            q2 = int(buy_cap * self.ACO_LADDER_FRACS[1])
            q3 = buy_cap - q1 - q2 # Remainder

            if q1 > 0: orders.append(Order(product, front_bid, q1))
            if q2 > 0: orders.append(Order(product, front_bid - 1, q2))
            if q3 > 0: orders.append(Order(product, front_bid - 2, q3))

        # Asks (Selling) Ladder
        if sell_cap > 0:
            q1 = int(sell_cap * self.ACO_LADDER_FRACS[0])
            q2 = int(sell_cap * self.ACO_LADDER_FRACS[1])
            q3 = sell_cap - q1 - q2

            if q1 > 0: orders.append(Order(product, front_ask, -q1))
            if q2 > 0: orders.append(Order(product, front_ask + 1, -q2))
            if q3 > 0: orders.append(Order(product, front_ask + 2, -q3))

        return orders

    def _ipr(self, product: str, od: OrderDepth, position: int, limit: int, prev_ema: float):
        orders  = []
        buy_cap = limit - position

        if od.buy_orders and od.sell_orders:
            current_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        elif od.sell_orders:
            current_mid = min(od.sell_orders) - 6.5
        elif od.buy_orders:
            current_mid = max(od.buy_orders) + 6.5
        else:
            current_mid = None

        alpha   = 0.2
        new_ema = current_mid if prev_ema is None else \
                  (current_mid * alpha + prev_ema * (1 - alpha)) \
                  if current_mid is not None else prev_ema

        fair_value = current_mid
        max_buy    = (fair_value + self.IPR_MAX_PREM) if fair_value is not None else float("inf")

        if buy_cap <= 0:
            return orders, new_ema

        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0 or ask > max_buy: break
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
                passive_bid = 13000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders, new_ema