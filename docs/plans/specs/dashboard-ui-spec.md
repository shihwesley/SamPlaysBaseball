---
name: dashboard-ui
phase: 3
sprint: 2
parent: api-layer
depends_on: [api-layer]
status: draft
created: 2026-02-16
---

# Dashboard UI Spec

Next.js pages and Plotly charts wrapping the 3D visualization and analysis results into a cohesive application.

## Requirements

- Pitcher profile pages with overview stats and baseline data
- Analysis results dashboard for all 6 modules
- Interactive charts: joint angle time series, release point heatmaps, fatigue curves
- Pitch-to-pitch comparison workflow
- Upload and processing status page
- Navigation and layout

## Acceptance Criteria

- [ ] Landing page: pitcher list with search/filter, last analyzed date
- [ ] Pitcher profile: name, pitch types, baseline stats, recent outings, arm slot
- [ ] Outing view: all pitches in an outing, color-coded by analysis flags
- [ ] Tipping dashboard: classifier accuracy, feature importance chart, per-pitch-type comparison
- [ ] Fatigue dashboard: metric curves over pitch count with changepoint markers
- [ ] Command dashboard: release point scatter plot with confidence ellipses by pitch type
- [ ] Arm slot dashboard: arm angle over time (per-pitch, per-outing, historical)
- [ ] Timing dashboard: kinetic chain sequence chart, hip-shoulder separation curve
- [ ] Baseline dashboard: radar chart of deviation across features, individual feature drill-downs
- [ ] Joint angle time series: Plotly line chart, aligned to foot plant, baseline envelope overlay
- [ ] Comparison page: select two pitches, see 3D side-by-side + feature diff table
- [ ] Upload page: drag-and-drop video, processing progress bar, result redirect
- [ ] Responsive layout: sidebar navigation, dark theme (matches baseball broadcast aesthetic)

## Technical Approach

Next.js 14+ with App Router. Plotly.js for 2D charts (better interactivity than static matplotlib). Tailwind CSS for styling. shadcn/ui for common components. Dark theme by default — player dev staff often work in dim video rooms.

Data fetching: server components for initial loads, client components for interactive charts. SWR or React Query for client-side data fetching with caching.

Layout: sidebar with pitcher list + nav, main content area with tab-based analysis views. The 3D viewer (from 3d-visualization spec) embeds as a component within pages.

## Files

| File | Purpose |
|------|---------|
| frontend/src/app/layout.tsx | Root layout, sidebar, dark theme |
| frontend/src/app/page.tsx | Landing page / pitcher list |
| frontend/src/app/pitcher/[id]/page.tsx | Pitcher profile |
| frontend/src/app/pitcher/[id]/outing/[outingId]/page.tsx | Outing analysis |
| frontend/src/app/compare/page.tsx | Pitch comparison |
| frontend/src/app/upload/page.tsx | Video upload |
| frontend/src/components/charts/AngleTimeSeries.tsx | Joint angle line chart |
| frontend/src/components/charts/ReleasePointScatter.tsx | Release point heatmap |
| frontend/src/components/charts/FatigueCurve.tsx | Fatigue metrics over pitch count |
| frontend/src/components/charts/KineticChain.tsx | Kinetic chain sequence chart |
| frontend/src/components/charts/RadarDeviation.tsx | Baseline deviation radar |
| frontend/src/components/charts/ArmSlotHistory.tsx | Arm slot over time |
| frontend/src/components/charts/TippingImportance.tsx | Feature importance bar chart |
| frontend/src/components/ui/Sidebar.tsx | Navigation sidebar |
| frontend/src/components/ui/PitcherCard.tsx | Pitcher list card |
| frontend/src/lib/api.ts | API client functions |

## Tasks

1. Set up Next.js project with Tailwind, dark theme, sidebar layout
2. Build pitcher list page with search/filter
3. Build pitcher profile page with overview stats
4. Build outing analysis page with pitch-by-pitch view
5. Implement Plotly chart components (angle time series, release scatter, fatigue curve, kinetic chain, radar, arm slot, tipping importance)
6. Build comparison page with feature diff table
7. Build upload page with drag-and-drop and progress bar
8. Integration: wire all pages to API endpoints

## Dependencies

- Upstream: api-layer
- Downstream: demo-mode
