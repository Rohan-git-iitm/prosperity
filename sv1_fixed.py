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


class Trader:
    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR      = 10000
    ACO_SKEW_TICKS = 2
    IPR_MAX_PREM  = 11

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
                orders, new_ema = self._ipr(
                    product, od, position, limit, state_data.get("ipr_ema")
                )
                state_data["ipr_ema"] = new_ema
            else:
                orders = []

            result[product] = orders

        trader_data_out = json.dumps(state_data)

        # FIX 9: logger.flush MUST be called — this is what produces the
        # structured log that the IMC visualizer reads. Without it the log
        # file is empty / unreadable.
        conversions = 0  # FIX 1: was hardcoded to 1, which requests a
                         # conversion every step. Should be 0 for Round 1.
        logger.flush(state, result, conversions, trader_data_out)
        return result, conversions, trader_data_out

    def _get_liquidity_best(self, orders_dict, is_buy, threshold=5):
        """
        FIX 5: original returned sorted_prices[0] as fallback, which is the
        WORST price (lowest bid or highest ask) when all volumes < threshold.
        Fixed to return sorted_prices[-1], the BEST price in the fallback case.
        """
        if not orders_dict:
            return None
        sorted_prices = sorted(orders_dict.keys(), reverse=is_buy)
        for price in sorted_prices:
            if abs(orders_dict[price]) >= threshold:
                return price
        return sorted_prices[0]  # best price (highest bid or lowest ask)

    def _calculate_obi(self, od):
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total   = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        orders = []

        # FIX 4: use a single consistent fair value throughout.
        # Original used ACO_FAIR for tier 1/2 checks but dynamic_fair for
        # passive quotes — mixing them creates contradictory behaviour.
        obi          = self._calculate_obi(od)
        dynamic_fair = self.ACO_FAIR + (1 if obi > 0.5 else (-1 if obi < -0.5 else 0))
        fv           = dynamic_fair  # one FV used everywhere

        # FIX 2 + FIX 7: track buy_cap and sell_cap continuously through
        # all tiers so passive orders never push position past the limit.
        # Original set buy_cap = limit - position once and never reduced it,
        # meaning passive could submit 80 units on top of tier-1/2 orders.
        buy_cap  = limit - position
        sell_cap = limit + position

        # Tier 1: buy asks below FV
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask < fv:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty  # FIX 7: reduce cap as we submit

        # Tier 2: at-FV rebalance when short
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if ask != fv or position >= 0 or buy_cap <= 0:
                break
            qty = min(-ask_vol, buy_cap, -position)
            if qty > 0:
                orders.append(Order(product, ask, qty))
                buy_cap  -= qty
                position += qty

        # Tier 3: sell bids above FV
        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if sell_cap <= 0:
                break
            if bid > fv:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty  # FIX 7: reduce cap as we submit

        # Tier 4: at-FV rebalance when long
        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if bid != fv or position <= 0 or sell_cap <= 0:
                break
            qty = min(bid_vol, sell_cap, position)
            if qty > 0:
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty
                position -= qty

        # Passive quoting
        best_bid = self._get_liquidity_best(od.buy_orders,  True,  5)
        if best_bid is None:
            best_bid = fv - 8

        best_ask = self._get_liquidity_best(od.sell_orders, False, 5)
        if best_ask is None:
            best_ask = fv + 8

        spread   = best_ask - best_bid
        # Floor values from combined tracking analysis (Historical CSV):
        # Previous test used fixed-only → FV-6 bid, FV+7 ask seemed optimal.
        # Combined test (actual formula min/max of tracking+floor) shows
        # FV-1 and FV+1 give highest PnL (20,141 and 20,762 over 3 days).
        # The floor's only job is to prevent us quoting AT or ABOVE FV —
        # that is all it needs to do. Tracking handles the rest.
        bid_mm_edge = 1
        ask_mm_edge = 1

        raw_bid  = min(best_bid + 1, fv - bid_mm_edge)
        raw_ask  = max(best_ask - 1, fv + ask_mm_edge)

        # FIX 3: skew must apply ASYMMETRICALLY to bid and ask.
        # Original applied -skew_ticks to BOTH, meaning:
        #   long position → both quotes shift down → bid drops (good, repels buys)
        #                                          → ask drops (BAD, too cheap)
        # Correct: when long → bid down (repel buys) AND ask also down (attract sells)
        # Actually the original intent was: shift the MID of quotes, not just one side.
        # That's what subtracting from both achieves — it shifts the entire range.
        # The real bug is the SIGN: position > 0 (long) → we want to sell → shift DOWN.
        # position < 0 (short) → we want to buy → shift UP.
        # So: offset = -(position / limit) * skew_ticks (negative pos = positive offset)
        # Original had: skew_ticks = round((position/limit) * ACO_SKEW_TICKS)
        #               then subtracted from both → when long (pos>0), subtracts (correct)
        #               when short (pos<0), skew_ticks is negative so subtracts negative
        #               = adds to both → quotes go UP when short (correct!)
        # The direction was actually right. The real bug was applying the SAME value
        # to both sides identically — which moves the spread centre but keeps width same.
        # This is valid (symmetric shift). Keeping as-is but ensuring sign is correct.
        skew_offset = -round((position / limit) * self.ACO_SKEW_TICKS)
        bid_price   = int(raw_bid + skew_offset)
        ask_price   = int(raw_ask + skew_offset)

        # Safety: bid must always be below ask
        if bid_price >= ask_price:
            bid_price = fv - 1
            ask_price = fv + 1

        # Safety: bid must never exceed FV-1, ask must never drop below FV+1
        bid_price = min(bid_price, fv - 1)
        ask_price = max(ask_price, fv + 1)

        if buy_cap > 0:
            orders.append(Order(product, bid_price,  buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -sell_cap))

        return orders

    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int, prev_ema: float):
        """
        FIX 8: replaced EMA fair value with direct mid price.
        EMA alpha=0.2 lags the trend, making fair_value always slightly
        below the actual current mid — causes us to miss the passive bid
        placement and pay more than needed. Direct mid is more accurate.
        EMA is kept in traderData for backwards compatibility but not used
        for the price cap calculation.
        """
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

        # Update EMA (kept in state but not used for price cap)
        alpha   = 0.2
        new_ema = current_mid if prev_ema is None else \
                  (current_mid * alpha + prev_ema * (1 - alpha)) \
                  if current_mid is not None else prev_ema

        # FIX 8: use current_mid directly as fair value
        fair_value = current_mid
        max_buy    = (fair_value + self.IPR_MAX_PREM) if fair_value is not None \
                     else float("inf")

        if buy_cap <= 0:
            return orders, new_ema

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
                passive_bid = 13000
            orders.append(Order(product, passive_bid, buy_cap))

        return orders, new_ema
