from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

# ═══════════════════════════════════════════════════════════════════
#  IMC Prosperity 4 – Round 1 Trader  (v2)
#
#  Three improvements over v1 based on CMU Physics team analysis:
#
#  [1] ACO: Exploit at-fair-value standing orders (NEW)
#        When a bot posts a bid/ask exactly at 10,000, use it to
#        rebalance inventory for free — no spread cost, no risk.
#        Applies only when it helps (long → sell at FV, short → buy).
#        Observed 2.1–2.2% of timesteps in historical data.
#
#  [2] ACO: Tighter passive spread  FV±1 instead of FV±2 (TIGHTENED)
#        Bot spread is ±8 from FV. Posting at ±1 makes us best price
#        in the book — bots fill us first. More fills = more spread.
#
#  [3] IPR: Real-time fair value + buy cap (IMPROVED)
#        Compute FV from order book mid each timestep.
#        Cap buys at FV+11 (max observed ask premium = 10.75).
#        Avoids the most expensive 5% of fills without missing trend.
# ═══════════════════════════════════════════════════════════════════

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    ACO_FAIR       = 10_000  # confirmed stable across all 3 historical days
    ACO_MM_EDGE    = 1       # [v2: tightened from 2]
    ACO_SKEW_TICKS = 2       # max inventory skew per side
    IPR_MAX_PREM   = 11      # [v2: never buy if ask > fair_value + this]

    def run(self, state: TradingState):
        result = {}
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

        return result, 0, ""

    # ════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM — 4-tier Market Making
    # ════════════════════════════════════════════════════════════
    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        """
        Tier 1  ask < FV           buy  (guaranteed profit)
        Tier 2  bid > FV           sell (guaranteed profit)
        Tier 3  ask == FV, pos<0   buy  to rebalance short to neutral [NEW]
        Tier 4  bid == FV, pos>0   sell to rebalance long  to neutral [NEW]
        Tier 5  passive FV±1 quotes with inventory skew               [TIGHTER]
        """
        orders   = []
        fv       = self.ACO_FAIR
        buy_cap  = limit - position
        sell_cap = limit + position

        # ── Tiers 1 & 3: buy-side takes ─────────────────────────
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask < fv:
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap  -= qty
            elif ask == fv and position < 0:
                # Rebalance short toward zero — don't overshoot into long
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap  -= qty
                    position += qty

        # ── Tiers 2 & 4: sell-side takes ────────────────────────
        for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
            if sell_cap <= 0:
                break
            if bid > fv:
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty
            elif bid == fv and position > 0:
                # Rebalance long toward zero — don't overshoot into short
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap  -= qty
                    position  -= qty

        # ── Tier 5: passive market-making quotes ────────────────
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

    # ════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — Trend Following (max long)
    # ════════════════════════════════════════════════════════════
    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        """
        Price drifts +1,000/day (+0.001 per timestamp unit).
        Hold maximum long (80) at all times — never sell.

        v2: compute real-time FV from order book mid, cap buys at FV+11.
        Historical data: ask avg = FV+6.5, max observed = FV+10.75.
        Residuals from trend are only ±2 std — mid IS fair value.
        """
        orders  = []
        buy_cap = limit - position

        if buy_cap <= 0:
            return orders

        # ── Real-time fair value from order book mid ─────────────
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

        # ── Hit asks up to price cap ─────────────────────────────
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0:
                break
            if ask > max_buy:
                break
            qty = min(-ask_vol, buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap -= qty

        # ── Passive bid for remaining capacity ───────────────────
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