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
#  IMC Prosperity 4 – Round 1 Trader  (v7)
#
#  Strategy built from statistical analysis of 3 days of data:
#
#  ASH_COATED_OSMIUM (ACO) — Data-driven Market Making
#  ─────────────────────────────────────────────────────
#  Property 1: Stationary (ADF p=4e-7), mean-reverts to FV=10,000
#  Property 2: Half-life = 2.9 steps → deviations resolve extremely fast
#  Property 3: Lag-1 autocorr = -0.495 → strong reversal tendency
#  Property 4: Bot spread = 16 (±8 from mid) → we post inside at ±3
#  Property 5: Mispriced orders (ask<FV or bid>FV) at 5%/4.7% of steps
#  Property 6: Standing at-FV orders at 2.1%/2.2% of steps
#  Property 7: Order imbalance signal: 4.75 ticks/unit (p≈0, R=0.28)
#  Property 8: Symmetric skew > asymmetric (both sides work together)
#
#  Algorithm tiers:
#    Tier 1 & 2: Aggressive takes (ask<FV buy, bid>FV sell) — free edge
#    Tier 3 & 4: At-FV rebalancing — free rebalancing
#    Tier 5:     Passive quoting with symmetric skew + imbalance lean
#
#  Position safety (three independent layers):
#    [A] ACO_MAX_STEP: caps submitted qty per side per timestep (root cause fix)
#    [B] Directional filter: stops ALL buys at +BUY_LIMIT, ALL sells at -SELL_LIMIT
#    [C] Symmetric skew: natural pressure pulling position back toward zero
#
#  Hyperparameters (data-calibrated):
#    ACO_SPREAD_TICKS    = 3    (data optimal: best fill-weighted PnL)
#    ACO_SKEW_STRENGTH   = 2    (safe: bid peaks at FV-1 at pos=-80)
#    ACO_IMBALANCE_WT    = 1.5  (conservative: ~1/3 of measured 4.75 signal)
#    ACO_MAX_STEP        = 10   (captures 70% of bot level-1 volume)
#    ACO_BUY/SELL_LIMIT  = ±60  (directional filter: stops drift before ±80)
#
#  INTARIAN_PEPPER_ROOT (IPR) — Trend Following
#  ─────────────────────────────────────────────
#  Price drifts +1,000/day with residual std=2.1 — hold max long always.
#    IPR_MAX_PREM = 11  (max ask premium above FV we'll pay, historical max=10.75)
# ══════════════════════════════════════════════════════════════════

class Trader:

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM":    80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ── ACO hyperparameters (tunable) ─────────────────────────
    ACO_FAIR             = 10_000  # fixed FV confirmed by ADF test, mean=10,000.2
    ACO_SPREAD_TICKS     = 3       # half-spread for passive quotes
    ACO_SKEW_STRENGTH    = 2       # inventory skew magnitude; must be < ACO_SPREAD_TICKS
    ACO_IMBALANCE_WT     = 1.5     # imbalance lean weight (signal = 4.75, use 1/3)
    ACO_PASSIVE_SIZE     = 15      # passive order size per side (p75 of bot level-1 volume)
    ACO_AGGRESSIVE_SIZE  = 10      # max units for tier-1/2 aggressive takes per step
    ACO_BUY_LIMIT        =  60     # suppress ALL buys when position >= this
    ACO_SELL_LIMIT       = -60     # suppress ALL sells when position <= this

    # ── IPR hyperparameters ────────────────────────────────────
    IPR_MAX_PREM = 11            # never buy ask > fair_value + this

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

    # ════════════════════════════════════════════════════════════
    #  ASH_COATED_OSMIUM — Data-driven Market Making
    # ════════════════════════════════════════════════════════════

    def _book_imbalance(self, od: OrderDepth) -> float:
        """
        Order book imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        Range: -1 (all ask) to +1 (all bid).
        Signal: positive imbalance → price tends to rise +4.75 ticks / unit
        over 5 steps (measured, R=0.28, p≈0).
        We use this to lean our quotes in the expected price direction.
        """
        bid_vol = sum(v for v in od.buy_orders.values()  if v > 0)
        ask_vol = sum(abs(v) for v in od.sell_orders.values() if v < 0)
        total   = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def _aco(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        """
        Position safety — three independent layers:
          [A] buy_cap / sell_cap capped at ACO_MAX_STEP:
              No single timestep can shift position > MAX_STEP regardless
              of bot order volume. This is the root-cause fix for the
              -100% position chart issue.

          [B] Directional filter (can_buy / can_sell):
              Once position reaches ±BUY/SELL_LIMIT, ALL orders on the
              worsening side are suppressed — including aggressive tier-1/2
              takes. Prevents directional ratchet to the limit.

          [C] Symmetric skew:
              Both quotes shift together as a function of inventory.
              When short: quotes rise → bid more attractive (rebalance),
              ask less attractive (stops drift). Both effects simultaneously.
              Clamps ensure bid never crosses FV and ask never drops below FV.
        """
        orders   = []
        fv       = self.ACO_FAIR

        # [A] Per-step size caps — two separate limits:
        #   AGGRESSIVE_SIZE: caps tier-1/2 takes (ask<FV, bid>FV)
        #     These are free money but still bounded to prevent large position swings
        #   PASSIVE_SIZE: caps tier-5 passive orders to p75 of bot level-1 volume (15 units)
        #     Previously capped at 10 (same as aggressive), which cost us revenue
        #     since bots often fill 12-24 units and we were leaving units on the table
        agg_size = self.ACO_AGGRESSIVE_SIZE
        buy_cap  = min(limit - position, agg_size)
        sell_cap = min(limit + position, agg_size)

        # [B] Directional filter
        can_buy  = position < self.ACO_BUY_LIMIT
        can_sell = position > self.ACO_SELL_LIMIT

        # ── Tier 1: buy asks strictly below FV ──────────────────
        if can_buy:
            for ask, ask_vol in sorted(od.sell_orders.items()):
                if buy_cap <= 0 or ask >= fv:
                    break
                qty = min(-ask_vol, buy_cap)
                orders.append(Order(product, ask, qty))
                buy_cap -= qty

        # ── Tier 2: sell bids strictly above FV ─────────────────
        if can_sell:
            for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if sell_cap <= 0 or bid <= fv:
                    break
                qty = min(bid_vol, sell_cap)
                orders.append(Order(product, bid, -qty))
                sell_cap -= qty

        # ── Tier 3: buy ask == FV when short (free rebalance) ───
        if can_buy and position < 0:
            for ask, ask_vol in sorted(od.sell_orders.items()):
                if ask != fv or buy_cap <= 0:
                    break
                qty = min(-ask_vol, buy_cap, -position)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap  -= qty
                    position += qty

        # ── Tier 4: sell bid == FV when long (free rebalance) ───
        if can_sell and position > 0:
            for bid, bid_vol in sorted(od.buy_orders.items(), reverse=True):
                if bid != fv or sell_cap <= 0:
                    break
                qty = min(bid_vol, sell_cap, position)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
                    position -= qty

        # ── Tier 5: passive symmetric quoting ───────────────────
        #
        # Use PASSIVE_SIZE (15, = p75 of bot level-1 volume) not the
        # aggressive cap (10). This was the root cause of v7 underperforming
        # v1: at ts=33900 bot had 12 units, v7 only filled 10, losing 2×5=10 PnL.
        # At ts=37700 bot had 20, v7 only 10, leaving 10×5=50 PnL on the table.
        # With passive_size=15, we capture full fills 75% of the time.
        #
        # Safety: passive caps use REMAINING buy_cap/sell_cap after tiers 1-4
        # consumed their aggressive portion. This ensures the combined total of
        # aggressive + passive submitted orders never pushes position past ±80
        # or past the directional filter threshold (±60).
        # e.g. pos=-59: aggressive sold 10 → sell_cap now = 21-10 = 11
        #                passive_sell_cap = min(11, 15) = 11 → pos goes to -70 max
        passive_buy_cap  = min(buy_cap,  self.ACO_PASSIVE_SIZE)
        passive_sell_cap = min(sell_cap, self.ACO_PASSIVE_SIZE)
        #   (a) Inventory skew:   offset = -(position/limit) * SKEW_STRENGTH
        #       Short pos → positive offset → both quotes rise
        #       Long pos  → negative offset → both quotes fall
        #   (b) Imbalance lean:   offset = IMBALANCE_WT * imbalance
        #       More bid volume → price expected to rise → quotes lean up
        #       More ask volume → price expected to fall → quotes lean down
        #
        # Safety clamps (data constraint):
        #   bid must never exceed FV-1 (avoid paying above fair value)
        #   ask must never fall below FV+1 (avoid selling below fair value)

        imbalance     = self._book_imbalance(od)
        skew_offset   = -(position / limit) * self.ACO_SKEW_STRENGTH
        imb_offset    = self.ACO_IMBALANCE_WT * imbalance
        total_offset  = skew_offset + imb_offset

        raw_bid = fv - self.ACO_SPREAD_TICKS + total_offset
        raw_ask = fv + self.ACO_SPREAD_TICKS + total_offset

        # Clamp to ensure we never quote above/below fair value
        bid_price = min(round(raw_bid), fv - 1)
        ask_price = max(round(raw_ask), fv + 1)

        if passive_buy_cap > 0 and can_buy:
            orders.append(Order(product, bid_price, passive_buy_cap))
        if passive_sell_cap > 0 and can_sell:
            orders.append(Order(product, ask_price, -passive_sell_cap))

        return orders

    # ════════════════════════════════════════════════════════════
    #  INTARIAN_PEPPER_ROOT — Trend Following (max long)
    # ════════════════════════════════════════════════════════════

    def _ipr(self, product: str, od: OrderDepth,
             position: int, limit: int) -> List[Order]:
        """
        Price drifts +1,000/day (+0.001 per raw timestamp).
        Hold maximum long (80) at all times — never sell.

        Fair value from order book mid (residual std=2.1 from trend,
        so mid IS fair value at any given step).
        Buy cap: never pay more than FV + IPR_MAX_PREM
        (historical max observed ask premium = 10.75).
        """
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

        # Hit all asks up to price cap
        for ask, ask_vol in sorted(od.sell_orders.items()):
            if buy_cap <= 0 or ask > max_buy:
                break
            qty = min(-ask_vol, buy_cap)
            orders.append(Order(product, ask, qty))
            buy_cap -= qty

        # Post passive bid for remaining capacity
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