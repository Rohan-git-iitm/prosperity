import json
from typing import Any, List, Dict

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
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {symbol: [order_depth.buy_orders, order_depth.sell_orders] for symbol, order_depth in order_depths.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        return [[trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp] for trade_list in trades.values() for trade in trade_list]

    def compress_observations(self, observations: Observation) -> list[Any]:
        return [observations.plainValueObservations, observations.conversionObservations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        return [[order.symbol, order.price, order.quantity] for order_list in orders.values() for order in order_list]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[:max_length - 3] + "..."

logger = Logger()

# ═══════════════════════════════════════════════════════════════════
#  Trading Algorithm v2 (Optimized)
# ═══════════════════════════════════════════════════════════════════

class Trader:
    def __init__(self):
        self.ipr_ema = None 
        self.osmium_ema = None

    def calculate_vwap(self, orders_dict: Dict[int, int]) -> float:
        if not orders_dict: return 0.0
        total_vol = sum(abs(v) for v in orders_dict.values())
        total_val = sum(p * abs(v) for p, v in orders_dict.items())
        return total_val / total_vol if total_vol > 0 else 0.0

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        
        # 1. LOAD MEMORY
        if state.traderData:
            try:
                memory = json.loads(state.traderData)
                self.ipr_ema = memory.get("ipr_ema", None)
                self.osmium_ema = memory.get("osmium_ema", None)
            except: pass

        for product, order_depth in state.order_depths.items():
            orders: List[Order] = []
            limit = 80
            pos = state.position.get(product, 0)
            
            # --- Robust Mid-Price Calculation ---
            v_bid = self.calculate_vwap(order_depth.buy_orders)
            v_ask = self.calculate_vwap(order_depth.sell_orders)
            
            if v_bid > 0 and v_ask > 0:
                mid = (v_bid + v_ask) / 2
            elif v_bid > 0: mid = v_bid
            elif v_ask > 0: mid = v_ask
            else: continue

            # ─────────────────────────────────────────────────────────────
            # STRATEGY 1: ASH_COATED_OSMIUM (The Pure Market Maker)
            # ─────────────────────────────────────────────────────────────
            if product == "ASH_COATED_OSMIUM":
                if self.osmium_ema is None: self.osmium_ema = mid
                else: self.osmium_ema = (0.4 * mid) + (0.6 * self.osmium_ema)

                # Skew quotes to keep inventory near 0
                skew = int((pos / limit) * 4) 
                
                # Resting orders around EMA
                orders.append(Order(product, int(self.osmium_ema) - 2 - skew, limit - pos))
                orders.append(Order(product, int(self.osmium_ema) + 2 - skew, -limit - pos))

            # ─────────────────────────────────────────────────────────────
            # STRATEGY 2: INTARIAN_PEPPER_ROOT (Aggressive Accumulator)
            # ─────────────────────────────────────────────────────────────
            elif product == "INTARIAN_PEPPER_ROOT":
                if self.ipr_ema is None: self.ipr_ema = mid
                else: self.ipr_ema = (0.1 * mid) + (0.9 * self.ipr_ema)

                # PHASE 1: Aggressive Taker (Get to 80 immediately)
                # This matches your original strategy's success
                buy_cap = limit - pos
                if buy_cap > 0:
                    for ask, vol in sorted(order_depth.sell_orders.items()):
                        # Only take if price isn't a massive outlier above trend
                        if ask <= self.ipr_ema + 15: 
                            take_qty = min(-vol, buy_cap)
                            orders.append(Order(product, ask, take_qty))
                            buy_cap -= take_qty
                            pos += take_qty
                        if buy_cap <= 0: break

                # PHASE 2: Passive Support
                # Place a big bid at the best available bid to stay at 80
                if buy_cap > 0:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else int(mid)
                    orders.append(Order(product, best_bid, buy_cap))

                # PHASE 3: Relief Valve
                # Only sell if we are full AND the price wiggles significantly above trend
                if pos > 70:
                    sell_price = int(mid) + 6
                    orders.append(Order(product, sell_price, -(pos - 70)))

            result[product] = orders

        # 2. SAVE MEMORY
        trader_data = json.dumps({"ipr_ema": self.ipr_ema, "osmium_ema": self.osmium_ema})
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data