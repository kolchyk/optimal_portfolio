# Dashboard Styling & Color Reference

## Streamlit Theme (`.streamlit/config.toml`)
- **Base:** dark
- **Primary:** `#00B4D8` (Cyan)
- **Background:** `#0A0E17` (Dark Navy)
- **Secondary BG:** `#111827` (Dark Gray-Blue)
- **Text:** `#E2E8F0` (Light Gray-Blue)
- **Font:** sans serif

## Fonts
- **Primary:** Inter (400, 500, 600, 700) — imported via Google Fonts
- **Monospace:** JetBrains Mono (400, 500) — for metric values, data tables
- **Fallback:** -apple-system, BlinkMacSystemFont, sans-serif

## Chart Colors (`dashboard.py`)
| Variable | Hex | Usage |
|----------|-----|-------|
| `COLOR_STRATEGY` | `#00B4D8` | Strategy line (Cyan) |
| `COLOR_BENCHMARK` | `#64748B` | S&P 500 / benchmark (Slate Gray) |
| `COLOR_POSITIVE` | `#10B981` | Positive returns (Green) |
| `COLOR_NEGATIVE` | `#EF4444` | Negative returns (Red) |
| `COLOR_DRAWDOWN` | `#F43F5E` | Drawdown fill (Rose Pink) |
| `COLOR_FRONTIER` | `#F59E0B` | Efficient frontier (Amber/Gold) |
| `COLOR_PORTFOLIO` | `#22D3EE` | Portfolio marker (Light Cyan) |

## Background Colors
| Hex | Usage |
|-----|-------|
| `#0A0E17` | Page background |
| `#111827` | Card background, range slider |
| `#0D1117` | Chart plot area |
| `#1E293B` | Chart grid lines, zeroline |
| `#1F2937` | Card borders, section dividers, axis lines |
| `#374151` | Metric card hover border |

## Text Colors
| Hex | Usage |
|-----|-------|
| `#E2E8F0` | Primary text, headings, metric values (neutral) |
| `#94A3B8` | Labels, legends, axis text, section headers |
| `#64748B` | Benchmark labels, muted text, annotations |
| `#475569` | Tooltip icons, hints |

## Plotly Base Layout
- **Template:** `plotly_dark`
- **Margins:** L50 R20 T50 B40
- **Paper BG:** transparent `rgba(0,0,0,0)`
- **Plot BG:** `#0D1117`
- **Font:** Inter, sans-serif, 12px, `#94A3B8`
- **Title:** 14px, `#E2E8F0`, left-aligned
- **Legend:** horizontal, bottom→top, 11px, `#94A3B8`, transparent BG
- **Grid/zeroline:** `#1E293B`; axis line: `#1F2937`

## Line Styles
- **Strategy:** solid, width 2.5
- **Benchmark:** dotted (`"dot"`), width 1.5
- **Drawdown strategy:** solid, width 1.5, fill `rgba(244,63,94,0.15)`
- **Drawdown benchmark:** dashed, width 1, fill `rgba(100,116,139,0.08)`
- **Rolling Sharpe strategy:** solid, width 1.5
- **Rolling Sharpe benchmark:** dashed, width 1
- **Reference Y=0:** dotted `#374151` width 1
- **Reference Y=1:** dashed `#1E293B` width 1, annotation "Good" `#64748B`

## Heatmap Color Scales
### Monthly Returns
```
[0, "#991B1B"]     → Dark Red (most negative)
[0.25, "#DC2626"]  → Red
[0.45, "#1E293B"]  → Dark Gray (neutral)
[0.55, "#1E293B"]  → Dark Gray (neutral)
[0.75, "#059669"]  → Dark Green
[1, "#047857"]     → Darker Green (most positive)
```
Text: 11px `#E2E8F0`, zmid=0

### Asset Return Distribution (Efficient Frontier page)
Same red-green scale but neutral band uses `#64748B` (Slate Gray) at 0.45–0.55.

## Marker Styles
- **Portfolio star:** size 16, `#22D3EE`, symbol `"star"`, line 2px `#0A0E17`
- **S&P 500 diamond:** size 14, `#64748B`, symbol `"diamond"`, line 2px `#0A0E17`
- **Frontier scatter:** size 6, line 0.5px `#475569`, opacity 0.8
- **Pie chart border:** `#0A0E17` width 2

## Metric Cards (CSS)
- **BG:** `#111827`, border 1px `#1F2937`, radius 6px
- **Padding:** 16px 20px, margin 4px 0
- **Hover:** border-color `#374151`
- **Label:** `#94A3B8`, 0.72rem, uppercase, 500 weight, 1px letter-spacing
- **Value:** 1.4rem, 700 weight, JetBrains Mono, -0.5px letter-spacing
- **Positive:** left border 3px `#10B981`, value color `#10B981`
- **Negative:** left border 3px `#EF4444`, value color `#EF4444`
- **Neutral:** left border 3px `#64748B`, value color `#E2E8F0`
- **Tooltip icon:** `#475569`, 0.65rem

## Section Headers (CSS)
- **Color:** `#94A3B8`, 0.75rem, uppercase, 1.5px letter-spacing, weight 600
- **Border-bottom:** 1px `#1F2937`
- **Divider:** border-top 1px `#1F2937`, margin 2rem 0 1.5rem

## Sidebar
- **Border-right:** 1px `#1F2937`
- **Run button:** gradient 135deg `#00B4D8` → `#0284C7`, uppercase, 600 weight, 0.85rem
- **Title:** `#E2E8F0`, 700 weight
- **Subtitle:** `#64748B`, 0.8rem

## Chart Heights
| Chart | Height |
|-------|--------|
| Equity Curve | 480px |
| Drawdown | 320px |
| Monthly Heatmap | dynamic (rows×40+80) |
| Yearly Returns | 350px |
| Rolling Sharpe | 320px |
| Allocation Pie | 450px |
| Efficient Frontier | 550px |
| Data Table | 400px |

## Layout
- Metric cards: 7 columns (Total Return, CAGR, Max DD, Sharpe, Calmar, Vol, Win Rate)
- Charts: full-width (`use_container_width=True`)
- Range slider on equity curve: BG `#111827`
