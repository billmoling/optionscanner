# WheelStrategy - Sell Put Scanner Design Specification

**Date:** 2026-04-08
**Status:** Draft
**Author:** Claude Code (AI Assistant)

---

## Overview

本设计文档描述了在 optionscanner 平台中新增 **WheelStrategy** 模块的完整规格，专门用于扫描和交易 **Cash-Secured Put（现金担保卖权）** 机会。该策略通过卖出虚值 Put 期权收取权利金，目标是在不被行权的前提下赚取时间价值衰减收益。

### 设计目标

1. **纯 Sell Put 策略** - 实现现金担保卖权（CSP）扫描，不自动行权持股
2. **超短期到期日** - 聚焦 0-15 天到期的期权，最大化时间价值衰减速率
3. **量化筛选** - 应用 IV Rank、成交量、年化 ROI、OTM 概率四重过滤
4. **概率分析** - 基于 Black-Scholes 模型计算不被行权概率
5. **可扩展扫描框架** - 模块化设计，支持未来添加其他扫描策略

### 非目标

- 完整的 Wheel 循环（持股后 Sell Covered Call）
- 自动行权或股票买入逻辑
- 修改现有 PutCreditSpreadStrategy
- 替换 IBKR 数据获取层

---

## 案例学习：AI 期权扫描最佳实践

基于用户分享的 AI Sell Put 扫描案例，提取以下核心特性：

| 特性 | 案例实现 | 本设计采用 |
|------|---------|-----------|
| 自选股扫描 | 12 只股票（NVDA、AAPL、SPY、TQQQ、SOXL 等） | 复用 config.yaml tickers |
| 到期日覆盖 | 0-45 天内 10 个日期 | 0-15 天，所有可用到期日 |
| IV 筛选 | IV ≥ 30% | IV Rank ≥ 30% |
| 成交量筛选 | 成交量 ≥ 500 张 | 成交量 ≥ 500 且 OI ≥ 1000 |
| 年化 ROI | 目标 ≥ 30% | 年化 ROI ≥ 30% |
| 概率计算 | Black-Scholes OTM 概率 ≥ 60% | 完全采用 |
| 输出格式 | Excel 4 工作表 | CSV + Slack 增强通知 |

---

## Architecture

### 模块依赖关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    WheelStrategy Module                          │
├─────────────────────────────────────────────────────────────────┤
│  strategy_wheel.py          │  核心策略实现                      │
│  scanners/put_scanner.py    │  通用扫描框架                      │
│  analytics/bs_model.py      │  Black-Scholes 概率计算            │
│  filters/options_filters.py │  IV/成交量/ROI筛选器               │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │     Existing Framework        │
                │  option_data.py               │
                │  strategies/base.py           │
                │  signal_ranking.py            │
                └───────────────────────────────┘
```

### 依赖顺序

```
1. analytics/bs_model.py      - 基础数学模型，无内部依赖
   ↓
2. filters/options_filters.py - 依赖 bs_model 计算概率
   ↓
3. scanners/put_scanner.py    - 依赖 filters 进行筛选
   ↓
4. strategy_wheel.py          - 依赖 scanner 生成信号
```

---

## Module 1: Black-Scholes 概率计算 (`analytics/bs_model.py`)

### 目的

实现 Black-Scholes 期权定价模型，计算期权的理论价格和不被行权概率。

### 组件

#### BSModel 类

```python
@dataclass(slots=True)
class BSModel:
    """Black-Scholes 期权定价和概率计算."""
    
    risk_free_rate: float = 0.05  # 无风险利率（年化）
    
    def calculate_d1_d2(
        self,
        S: float,  # 标的价格
        K: float,  # 行权价
        T: float,  # 到期时间（年）
        sigma: float,  # 隐含波动率
    ) -> Tuple[float, float]:
        """计算 d1 和 d2 参数."""
        
    def calculate_option_price(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,  # "CALL" or "PUT"
    ) -> float:
        """计算期权理论价格."""
        
    def calculate_otm_probability(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
    ) -> float:
        """
        计算期权不被行权（保持虚值）的概率.
        
        对于 Put 期权：P(股价 > 行权价) = N(d2)
        对于 Call 期权：P(股价 < 行权价) = N(-d2)
        """
```

### 配置参数

```yaml
# config.yaml
black_scholes:
  risk_free_rate: 0.05  # 5% 无风险利率
  probability_threshold: 0.60  # OTM 概率 ≥ 60%
```

---

## Module 2: 期权筛选器 (`filters/options_filters.py`)

### 目的

实现可组合的期权筛选管道，支持多个筛选条件的灵活配置。

### 组件

#### FilterResult 数据类

```python
@dataclass(slots=True)
class FilterResult:
    """筛选结果."""
    
    passed: bool
    reason: str
    metrics: Dict[str, Any]  # 包含 IV、volume、ROI、probability 等
```

#### 筛选器接口

```python
class OptionFilter(abc.ABC):
    """期权筛选器基类."""
    
    @abc.abstractmethod
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        """检查期权是否通过筛选."""
```

#### 具体筛选器实现

```python
@dataclass(slots=True)
class IVRankFilter(OptionFilter):
    """IV Rank 筛选器."""
    min_iv_rank: float = 0.30  # ≥ 30%
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        iv_rank = option_data.get("iv_rank", 0.0)
        if iv_rank >= self.min_iv_rank:
            return FilterResult(passed=True, reason=f"IV Rank {iv_rank:.1%} >= {self.min_iv_rank:.0%}")
        return FilterResult(passed=False, reason=f"IV Rank {iv_rank:.1%} < {self.min_iv_rank:.0%}")


@dataclass(slots=True)
class VolumeFilter(OptionFilter):
    """成交量和持仓量筛选器."""
    min_volume: int = 500
    min_open_interest: int = 1000
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        volume = option_data.get("volume", 0)
        oi = option_data.get("open_interest", 0)
        if volume >= self.min_volume and oi >= self.min_open_interest:
            return FilterResult(passed=True, reason=f"Volume {volume} >= {self.min_volume}, OI {oi} >= {self.min_open_interest}")
        return FilterResult(passed=False, reason=f"Liquidity too low (vol={volume}, oi={oi})")


@dataclass(slots=True)
class AnnualizedROIFilter(OptionFilter):
    """年化 ROI 筛选器."""
    min_annualized_roi: float = 0.30  # ≥ 30%
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        annualized_roi = option_data.get("annualized_roi", 0.0)
        if annualized_roi >= self.min_annualized_roi:
            return FilterResult(passed=True, reason=f"Annualized ROI {annualized_roi:.1%} >= {self.min_annualized_roi:.0%}")
        return FilterResult(passed=False, reason=f"Annualized ROI {annualized_roi:.1%} < {self.min_annualized_roi:.0%}")


@dataclass(slots=True)
class OTMProbabilityFilter(OptionFilter):
    """OTM 概率筛选器（基于 Black-Scholes）."""
    min_otm_probability: float = 0.60  # ≥ 60%
    bs_model: BSModel = field(default_factory=BSModel)
    
    def check(self, option_data: Dict[str, Any], context: Dict[str, Any]) -> FilterResult:
        S = context["underlying_price"]
        K = option_data["strike"]
        T = option_data["days_to_expiry"] / 365.0
        sigma = option_data.get("implied_volatility", 0.3)
        
        prob = self.bs_model.calculate_otm_probability(S, K, T, sigma, "PUT")
        if prob >= self.min_otm_probability:
            return FilterResult(passed=True, reason=f"OTM Probability {prob:.1%} >= {self.min_otm_probability:.0%}")
        return FilterResult(passed=False, reason=f"OTM Probability {prob:.1%} < {self.min_otm_probability:.0%}")
```

#### 筛选器管道

```python
@dataclass(slots=True)
class OptionFilterPipeline:
    """筛选器管道，按顺序执行多个筛选器."""
    
    filters: List[OptionFilter]
    
    def add_filter(self, filter: OptionFilter) -> None:
        self.filters.append(filter)
    
    def check_all(
        self,
        option_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        执行所有筛选器.
        
        Returns:
            (passed, reasons): 是否通过所有筛选，以及各筛选器的原因列表
        """
        reasons = []
        for f in self.filters:
            result = f.check(option_data, context)
            reasons.append(result.reason)
            if not result.passed:
                return False, reasons
        return True, reasons
```

---

## Module 3: Put Scanner 扫描器 (`scanners/put_scanner.py`)

### 目的

扫描所有可用期权合约，应用筛选管道，返回符合条件的 Sell Put 机会。

### 组件

#### PutScanResult 数据类

```python
@dataclass(slots=True)
class PutScanResult:
    """Sell Put 扫描结果."""
    
    symbol: str
    expiry: datetime
    strike: float
    option_type: str  # "PUT"
    
    # 市场数据
    underlying_price: float
    option_bid: float
    iv_rank: float
    volume: int
    open_interest: int
    
    # 计算指标
    days_to_expiry: int
    annualized_roi: float
    otm_probability: float
    delta: float
    
    # 筛选状态
    passed_filters: bool
    filter_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于输出或进一步处理."""
```

#### PutScanner 类

```python
@dataclass(slots=True)
class PutScanner:
    """Sell Put 机会扫描器."""
    
    filter_pipeline: OptionFilterPipeline
    bs_model: BSModel = field(default_factory=BSModel)
    
    # 配置参数
    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 15
    min_strike_range_pct: float = 0.05  # 行权价需在标的价格下方 5% 以内
    max_strike_range_pct: float = 0.20  # 行权价需在标的价格下方 20% 以外
    
    def scan(
        self,
        snapshot: OptionChainSnapshot,
    ) -> List[PutScanResult]:
        """
        扫描单个股票的 Sell Put 机会.
        
        Args:
            snapshot: 期权链快照
        
        Returns:
            符合条件的 Sell Put 机会列表
        """
        results = []
        underlying_price = snapshot.underlying_price
        
        # 过滤 Put 期权
        puts = [opt for opt in snapshot.options if opt.get("option_type") == "PUT"]
        
        for option in puts:
            # 计算到期天数
            expiry = pd.to_datetime(option["expiry"], utc=True)
            days_to_expiry = (expiry - datetime.now(timezone.utc)).days
            
            # 跳过不符合到期日范围的期权
            if days_to_expiry < self.min_days_to_expiry or days_to_expiry > self.max_days_to_expiry:
                continue
            
            # 计算行权价范围（虚值 Put：行权价 < 标的价格）
            strike = float(option["strike"])
            otm_pct = (underlying_price - strike) / underlying_price
            if otm_pct < self.min_strike_range_pct or otm_pct > self.max_strike_range_pct:
                continue
            
            # 计算年化 ROI
            premium = float(option.get("bid", option.get("mark", 0.0)))
            collateral = strike  # 现金担保需要行权价金额
            roi = premium / collateral if collateral > 0 else 0.0
            annualized_roi = roi * (365.0 / max(days_to_expiry, 1))
            
            # 准备筛选上下文
            context = {
                "underlying_price": underlying_price,
                "symbol": snapshot.symbol,
            }
            
            # 更新 option 数据
            option["days_to_expiry"] = days_to_expiry
            option["annualized_roi"] = annualized_roi
            
            # 执行筛选管道
            passed, reasons = self.filter_pipeline.check_all(option, context)
            
            if passed:
                # 计算 OTM 概率
                T = days_to_expiry / 365.0
                sigma = option.get("implied_volatility", 0.3)
                otm_prob = self.bs_model.calculate_otm_probability(
                    underlying_price, strike, T, sigma, "PUT"
                )
                
                results.append(PutScanResult(
                    symbol=snapshot.symbol,
                    expiry=expiry,
                    strike=strike,
                    option_type="PUT",
                    underlying_price=underlying_price,
                    option_bid=premium,
                    iv_rank=option.get("iv_rank", 0.0),
                    volume=option.get("volume", 0),
                    open_interest=option.get("open_interest", 0),
                    days_to_expiry=days_to_expiry,
                    annualized_roi=annualized_roi,
                    otm_probability=otm_prob,
                    delta=option.get("delta", 0.0),
                    passed_filters=True,
                    filter_reasons=reasons,
                ))
        
        # 按年化 ROI 降序排序
        results.sort(key=lambda r: r.annualized_roi, reverse=True)
        return results
```

---

## Module 4: WheelStrategy 策略实现 (`strategies/strategy_wheel.py`)

### 目的

将 Put Scanner 集成到策略框架中，生成可执行的 TradeSignal。

### 配置参数

```yaml
# config.yaml
strategies:
  WheelStrategy:
    params:
      # 到期日范围
      min_days_to_expiry: 0
      max_days_to_expiry: 15
      
      # 筛选条件
      min_iv_rank: 0.30
      min_volume: 500
      min_open_interest: 1000
      min_annualized_roi: 0.30
      min_otm_probability: 0.60
      
      # 行权价范围
      min_strike_range_pct: 0.05
      max_strike_range_pct: 0.20
      
      # 其他
      max_signals_per_symbol: 3  # 每个股票最多输出 3 个信号
      published_win_rate: 0.65
```

### 策略实现

```python
from optionscanner.strategies.base import BaseOptionStrategy, TradeSignal, SignalLeg
from optionscanner.scanners.put_scanner import PutScanner, PutScanResult
from optionscanner.filters.options_filters import (
    OptionFilterPipeline,
    IVRankFilter,
    VolumeFilter,
    AnnualizedROIFilter,
    OTMProbabilityFilter,
)
from optionscanner.analytics.bs_model import BSModel


class WheelStrategy(BaseOptionStrategy):
    """Cash-Secured Put 策略.
    
    卖出虚值 Put 期权，收取权利金，目标是在不被行权的前提下
    赚取时间价值衰减收益。
    """
    
    def __init__(
        self,
        min_days_to_expiry: int = 0,
        max_days_to_expiry: int = 15,
        min_iv_rank: float = 0.30,
        min_volume: int = 500,
        min_open_interest: int = 1000,
        min_annualized_roi: float = 0.30,
        min_otm_probability: float = 0.60,
        min_strike_range_pct: float = 0.05,
        max_strike_range_pct: float = 0.20,
        max_signals_per_symbol: int = 3,
        published_win_rate: float = 0.65,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.max_signals_per_symbol = max_signals_per_symbol
        self.published_win_rate = published_win_rate
        
        # 构建筛选器管道
        filters = [
            IVRankFilter(min_iv_rank=min_iv_rank),
            VolumeFilter(min_volume=min_volume, min_open_interest=min_open_interest),
            AnnualizedROIFilter(min_annualized_roi=min_annualized_roi),
            OTMProbabilityFilter(min_otm_probability=min_otm_probability),
        ]
        self.filter_pipeline = OptionFilterPipeline(filters=filters)
        
        # 初始化扫描器
        self.scanner = PutScanner(
            filter_pipeline=self.filter_pipeline,
            min_days_to_expiry=min_days_to_expiry,
            max_days_to_expiry=max_days_to_expiry,
            min_strike_range_pct=min_strike_range_pct,
            max_strike_range_pct=max_strike_range_pct,
        )
    
    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        """处理期权链数据，生成 Sell Put 信号."""
        signals: List[TradeSignal] = []
        
        for snapshot in data:
            # 扫描 Sell Put 机会
            results = self.scanner.scan(snapshot)
            
            # 限制每个股票的信号数量
            for result in results[: self.max_signals_per_symbol]:
                signal = self._build_signal(result)
                if signal:
                    signals.append(signal)
        
        return signals
    
    def _build_signal(self, result: PutScanResult) -> TradeSignal:
        """构建 TradeSignal."""
        rationale = (
            f"Cash-Secured Put: sell {result.strike:.2f}P exp {result.expiry.strftime('%Y-%m-%d')} "
            f"premium ${result.option_bid:.2f} | "
            f"IV Rank {result.iv_rank:.1%} | Volume {result.volume} | OI {result.open_interest} | "
            f"Annualized ROI {result.annualized_roi:.1%} | "
            f"OTM Probability {result.otm_probability:.1%} | "
            f"DTE {result.days_to_expiry}d"
        )
        
        return TradeSignal(
            symbol=result.symbol,
            expiry=result.expiry,
            strike=result.strike,
            option_type="PUT",
            direction="SHORT_PUT",
            rationale=rationale,
            risk_reward_ratio=result.annualized_roi,  # 用年化 ROI 作为 R/R 代理
            max_profit=result.option_bid,
            max_loss=result.strike - result.option_bid,  # 行权价 - 权利金
            legs=(
                SignalLeg(
                    action="SELL",
                    option_type="PUT",
                    strike=result.strike,
                    expiry=result.expiry,
                    quantity=1,
                ),
            ),
        )
```

---

## 输出与通知增强

### CSV 输出字段

在现有 `signals_*.csv` 基础上新增字段：

| 字段 | 说明 |
|------|------|
| `iv_rank` | IV 百分位 |
| `volume` | 成交量 |
| `open_interest` | 持仓量 |
| `annualized_roi` | 年化收益率 |
| `otm_probability` | OTM 概率 |
| `days_to_expiry` | 到期天数 |
| `option_bid` | 权利金 |

### Slack 通知增强

在 `notifications/slack.py` 中新增 `send_wheel_strategy_signals` 方法：

```python
def send_wheel_strategy_signals(
    self,
    scan_results: List[PutScanResult],
    csv_path: Optional[Path] = None,
) -> None:
    """发送 WheelStrategy Sell Put 机会到 Slack."""
    if not self.enabled or not scan_results:
        return
    
    lines = [f"*{self.title}* | Sell Put Opportunities | {timestamp}"]
    lines.append("")
    
    # Top 5 推荐
    lines.append("*Top 5 Recommendations:*")
    for idx, result in enumerate(scan_results[:5], start=1):
        lines.append(
            f"*{idx}. {result.symbol}* | Strike ${result.strike:.0f} | "
            f"DTE {result.days_to_expiry}d | "
            f"ROI {result.annualized_roi:.1%} | "
            f"Prob {result.otm_probability:.0%}"
        )
    
    if csv_path:
        lines.append("")
        lines.append(f"Full results: `{csv_path}`")
    
    message = "\n".join(lines)
    # 发送 Slack...
```

---

## 配置集成

### config.yaml 新增配置

```yaml
strategies:
  WheelStrategy:
    params:
      min_days_to_expiry: 0
      max_days_to_expiry: 15
      min_iv_rank: 0.30
      min_volume: 500
      min_open_interest: 1000
      min_annualized_roi: 0.30
      min_otm_probability: 0.60
      min_strike_range_pct: 0.05
      max_strike_range_pct: 0.20
      max_signals_per_symbol: 3
    published_win_rate: 0.65

# Black-Scholes 配置
black_scholes:
  risk_free_rate: 0.05
  probability_threshold: 0.60
```

---

## 测试策略

### 单元测试

1. **BSModel 测试** (`tests/test_bs_model.py`)
   - 验证 d1/d2 计算
   - 验证期权价格计算（与已知参考值对比）
   - 验证 OTM 概率计算

2. **筛选器测试** (`tests/test_options_filters.py`)
   - 每个筛选器的边界条件测试
   - 筛选器管道组合测试

3. **扫描器测试** (`tests/test_put_scanner.py`)
   - 使用模拟期权链数据
   - 验证筛选结果正确性

4. **策略测试** (`tests/test_wheel_strategy.py`)
   - 集成测试，验证完整流程

### 集成测试

- 使用 IBKR 模拟账户数据
- 验证真实市场数据下的扫描结果

---

## 实施计划

### Phase 1: 基础模型
1. 实现 `analytics/bs_model.py`
2. 编写单元测试

### Phase 2: 筛选器
1. 实现 `filters/options_filters.py`
2. 编写单元测试

### Phase 3: 扫描器
1. 实现 `scanners/put_scanner.py`
2. 编写集成测试

### Phase 4: 策略集成
1. 实现 `strategies/strategy_wheel.py`
2. 集成到 runner.py
3. 端到端测试

### Phase 5: 输出与通知
1. 增强 CSV 输出字段
2. 增强 Slack 通知
3. 用户验收测试

---

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| IV 数据不可用 | 无法计算 IV Rank | 使用历史波动率替代，或跳过该筛选 |
| Black-Scholes 计算误差 | 概率不准确 | 与已知参考值验证，添加误差容忍度 |
| 超短期期权流动性差 | 无法成交 | 严格成交量/OI 筛选，设置最小流动性门槛 |
| 早于 0 天到期 | 已到期期权 | 严格过滤 days_to_expiry > 0 |

---

## 成功标准

1. **功能完整性**
   - [ ] 所有 4 个筛选器正常工作
   - [ ] Black-Scholes 概率计算误差 < 5%
   - [ ] 扫描结果按年化 ROI 排序

2. **性能指标**
   - [ ] 单次扫描（12 只股票 × 10 个到期日）< 5 秒
   - [ ] 内存占用 < 100MB

3. **用户体验**
   - [ ] Slack 通知显示 Top 5 推荐
   - [ ] CSV 输出包含所有关键字段
   - [ ] 配置参数可在 config.yaml 中调整

---

## 未来扩展

1. **完整 Wheel 循环** - 被行权后自动切换 Covered Call 模式
2. **动态调整** - 根据市场状态调整筛选门槛
3. **回测支持** - 历史数据回测框架
4. **自动交易** - 与 TradeExecutor 集成，自动提交订单

---

## 附录：Black-Scholes 公式参考

### d1 和 d2 计算

$$d1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d2 = d1 - \sigma\sqrt{T}$$

### Put 期权理论价格

$$P = K e^{-rT} N(-d2) - S N(-d1)$$

### Put 期权 OTM 概率

$$P(\text{OTM}) = P(S_T > K) = N(d2)$$

其中：
- $S$ = 标的价格
- $K$ = 行权价
- $r$ = 无风险利率
- $\sigma$ = 隐含波动率
- $T$ = 到期时间（年）
- $N(x)$ = 标准正态分布累积分布函数
