// CPU proof of the index arithmetic of the moe8 wide launch (5..8 rows in one fused launch): row windows of the front end, units of <= MAXR rows, shared-expert units,
// seldbg regions, workspace capacity. The kernel and this file call the SAME helpers (exl3_moe_fused_idx.h: win_*, build_units_wide, shared_p0, shared_rows, sel_wt_base,
// units_shared); allocation sizes are written out independently here. Run with -fsanitize=address,undefined so that any array overflow aborts.
// Build: g++ -O1 -g -std=c++17 -fsanitize=address,undefined -I<quant dir> moe8_bounds.cpp -o moe8_bounds [-DMUT=n]. Exit 0 = all pass.
// MUT: 1 groups cut every 5 rows, 2 urows array one entry short (TA + 1), 3 workspace sized for MAXR rows, 4 shared unit p0 without + MAXR * k, 5 window base dropped for r0 > 0,
//      6 seldbg weights at 64 for the wide shape, 7 window rows not clamped to MAXR, 8 gstart array two entries short (TA), 9 shared units = 1 (R rows in one unit)
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include "exl3_moe_fused_idx.h"
#ifndef MUT
#define MUT 0
#endif
using namespace mf;
static long long checks = 0, fails = 0;
#define CHK(c, ...) do { ++checks; if (!(c)) { if (fails < 12) { printf("FAIL " __VA_ARGS__); printf("\n"); } ++fails; } } while (0)

// helpers as called by the kernel, with the mutants applied on top
static int m_units_shared(int R) { return MUT == 9 ? 1 : units_shared(R); }
static int m_shared_rows(int R, int k) { return MUT == 9 ? R : shared_rows(R, k); }
static int m_shared_p0(int NA, int k) { return MUT == 4 ? NA : shared_p0(NA, k); }
static long m_win(long v, int r0) { return (MUT == 5 && r0 > 0) ? 0 : v; }
static int m_win_rows(int R, int r0) { return MUT == 7 ? R - r0 : win_rows(R, r0); }
static int m_sel_wt_base(int tr) { return MUT == 6 ? 64 : sel_wt_base(tr); }
static int build(const std::vector<int>& head, int NA, int R, std::vector<int>& gstart, std::vector<int>& urows)
{
    if (MUT == 1)
    {
        int g = 0, gsx = 0;
        for (int q = 0; q < NA; ++q) { if (head[q]) gsx = q; if (((q - gsx) % 5) == 0) gstart[g++] = q; }
        gstart[g] = NA;
        for (int u = 0; u < g; ++u) urows[u] = gstart[u + 1] - gstart[u];
        for (int k = 0; k < units_shared(R); ++k) urows[g + k] = shared_rows(R, k);
        return g;
    }
    return build_units_wide(head.data(), NA, R, gstart.data(), urows.data());
}

template <class S> void prove(const char* name)
{
    using DM = Dm<S>;
    printf("shape %s\n", name);
    const int TOPK = S::TOPK, NEXP = S::NEXP;
    // independent workspace capacity (elements), as exl3_moe_fused.cu sizes the buffers for 8 rows (MUT 3: 4 rows)
    const int CAPR = MUT == 3 ? 4 : 8;
    const int NARM = CAPR * TOPK + CAPR;
    const long cap_xin = 8L * S::H * S::D, cap_dots = (long) CAPR * (DM::MR + 1) * S::H, cap_post = (long) CAPR * S::H, cap_mixed = (long) CAPR * S::D,
               cap_scores = (long) CAPR * NEXP, cap_sgl = 64, cap_gu = 2L * NARM * S::INTER, cap_dn = (long) NARM * S::D, cap_ydbg = (long) CAPR * S::D;
    srand(77);
    for (int R = 1; R <= 8; ++R)
    {
        const int NA = R * TOPK, NAR = NA + R;
        const bool wide = R > MAXR;
        // ---- windows: rows [r0, r0 + rows) of the launch, each at most MAXR, covering [0, R) once; every window-shifted buffer index stays inside its buffer
        {
            std::vector<int> cov(R, 0);
            for (int r0 = 0; r0 < R; r0 += MAXR)
            {
                const int rows = m_win_rows(R, r0);
                CHK(rows >= 1 && rows <= MAXR, "R=%d r0=%d window rows %d", R, r0, rows);
                for (int r = 0; r < rows && r0 + r < R; ++r) cov[r0 + r]++;
                // largest local row = rows - 1; indices as the phases build them (xin_row, dots_idx, post r*H+h, mixed_row, scores_idx, sgl[r], xout_idx)
                const int rl = rows - 1;
                const long xin = m_win(win_xin<S>(r0), r0), dots = m_win(win_dots<S>(r0), r0), post = m_win(win_post<S>(r0), r0), mixed = m_win(win_mixed<S>(r0), r0),
                           scores = m_win(win_scores<S>(r0), r0), sgl = m_win(win_sgl(r0), r0);
                CHK(xin + xin_row<S>(rl, S::H - 1) + S::D <= cap_xin && xin + xout_idx<S>(rl, S::H - 1, S::D - 1) < cap_xin, "R=%d r0=%d xin", R, r0);
                CHK(dots + dots_idx<S>(rl, DM::MR, S::H - 1) < cap_dots, "R=%d r0=%d dots", R, r0);
                CHK(post + (long) rl * S::H + S::H - 1 < cap_post, "R=%d r0=%d post", R, r0);
                CHK(mixed + mixed_row<S>(rl) + S::D <= cap_mixed, "R=%d r0=%d mixed", R, r0);
                CHK(scores + scores_idx<S>(rl, NEXP - 1) < cap_scores, "R=%d r0=%d scores", R, r0);
                CHK(sgl + rl < cap_sgl, "R=%d r0=%d sgl", R, r0);
                // the window must also map onto the right rows: row r0 + r of the launch sits at the row-major position of row r0 + r
                CHK(win_xin<S>(r0) == xin_row<S>(r0, 0) && win_dots<S>(r0) == dots_idx<S>(r0, 0, 0) && win_mixed<S>(r0) == mixed_row<S>(r0) && win_scores<S>(r0) == scores_idx<S>(r0, 0), "R=%d r0=%d window base differs from the row base", R, r0);
                CHK(m_win(win_xin<S>(r0), r0) == xin_row<S>(r0, 0), "R=%d r0=%d window base (as used) differs from the row base", R, r0);
            }
            for (int r = 0; r < R; ++r) CHK(cov[r] == 1, "R=%d row %d covered %d times by the windows", R, r, cov[r]);
        }
        // ---- seldbg regions
        {
            const int wb = m_sel_wt_base(wide ? 8 : 4);
            CHK(NA <= wb || !wide || true, "");
            CHK(NA <= wb && wb + NA <= STAMP_BASE && wb + NA <= SELDBG_INTS, "R=%d seldbg: sel [0,%d) weights [%d,%d) overlap or reach the stamps", R, NA, wb, wb + NA);
        }
        if (!wide) continue;   // the unit builder is the wide instantiation only
        // ---- routings
        for (int mode = 0; mode < 4; ++mode)
            for (int trial = 0; trial < (mode == 3 ? 400 : 60); ++trial)
            {
                std::vector<int> sel(NA);
                for (int r = 0; r < R; ++r)
                {
                    std::vector<int> pick;
                    const int pool = mode == 2 ? TOPK + 1 + trial % 6 : (mode == 3 ? NEXP : NEXP);
                    const int base = mode == 1 ? 0 : (mode == 0 ? r * TOPK : (rand() % (NEXP - pool + 1)));
                    while ((int) pick.size() < TOPK)
                    {
                        int e = mode == 0 ? (base + (int) pick.size()) % NEXP : (mode == 1 ? (int) pick.size() : base + rand() % pool);
                        if (mode == 3 && trial % 2) e = rand() % 12;
                        if (mode == 3 && trial % 2 == 0) e = rand() % NEXP;
                        if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
                    }
                    for (int k = 0; k < TOPK; ++k) sel[r * TOPK + k] = pick[k];
                }
                // tail of the kernel: rank sort (expert ascending, slot ascending), head flags
                std::vector<int> se(NA), sa(NA), tok(NA), head(NA);
                for (int a = 0; a < NA; ++a) { int pp = 0; for (int b = 0; b < NA; ++b) pp += (sel[b] < sel[a]) || (sel[b] == sel[a] && b < a); se[pp] = sel[a]; sa[pp] = a; tok[pp] = a / TOPK; }
                for (int q = 0; q < NA; ++q) head[q] = (q == 0 || se[q - 1] != se[q]) ? 1 : 0;
                // arrays exactly as Smem declares them for TR = 8: gstart TA + 2, urows TA + 2 (MUT 2: TA + 1, MUT 8: gstart TA)
                const int TA = 8 * TOPK;
                std::vector<int> gstart(MUT == 8 ? TA : TA + 2, -1), urows(MUT == 2 ? TA + 1 : TA + 2, -1);
                const int ng = build(head, NA, R, gstart, urows);
                const int nsh = m_units_shared(R), NU = ng + nsh;
                CHK(NU <= TA + 2 && ng <= NA, "R=%d units %d exceed the arrays", R, NU);
                // routed units: rows <= MAXR, consecutive, same expert, cover [0, NA) once, token valid
                int next = 0;
                std::vector<int> seen(NA, 0);
                std::vector<int> outrow(NAR, 0);
                for (int u = 0; u < ng; ++u)
                {
                    const int p0 = gstart[u], rows = std::min(gstart[u + 1] - p0, MAXR);
                    CHK(urows[u] == rows && rows >= 1 && rows <= MAXR, "R=%d unit %d rows %d (urows %d)", R, u, rows, urows[u]);
                    CHK(p0 == next && gstart[u + 1] - gstart[u] == urows[u], "R=%d unit %d not consecutive / cut wrong (p0 %d next %d len %d rows %d)", R, u, p0, next, gstart[u + 1] - gstart[u], urows[u]);
                    next = p0 + urows[u];
                    for (int row = 0; row < urows[u]; ++row)
                    {
                        const int q = p0 + row;
                        CHK(q < NA && se[q] == se[p0], "R=%d unit %d row %d different expert", R, u, row);
                        CHK(tok[q] >= 0 && tok[q] < R, "R=%d token", R);
                        if (q < NA) { seen[q]++; outrow[q]++; }
                        // LDS: A / A2 hold MAXR rows of D (gate/up) or INTER (down): row index < MAXR
                        CHK(row < MAXR && (long) a_row_gu<S>(row, DM::CMB_BLOCKS - 1) + 128 <= (long) MAXR * S::D, "R=%d A buffer row %d", R, row);
                    }
                    // gate/up and down output rows of this unit inside [0, NAR) of the gu / dn buffers
                    CHK(gu_out<S>(1, NAR, p0) + (long) urows[u] * S::INTER <= (long) 2 * NAR * S::INTER && (long) dn_row<S>(p0) + (long) urows[u] * S::D <= (long) NAR * S::D, "R=%d unit %d output rows", R, u);
                }
                CHK(next == NA, "R=%d units cover %d of %d slots", R, next, NA);
                for (int q = 0; q < NA; ++q) CHK(seen[q] == 1, "R=%d slot %d covered %d times", R, q, seen[q]);
                // the (token, expert) pairs of the units equal the selection
                {
                    std::vector<std::pair<int, int>> a, b;
                    for (int q = 0; q < NA; ++q) a.push_back({tok[q], se[q]});
                    for (int x = 0; x < NA; ++x) b.push_back({x / TOPK, sel[x]});
                    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                    CHK(a == b, "R=%d (token, expert) pairs differ", R);
                }
                // shared units: rows 0..R-1 once, p0 in [NA, NAR), tokens p0 - NA + row
                std::vector<int> shseen(R, 0);
                for (int k = 0; k < nsh; ++k)
                {
                    const int p0 = m_shared_p0(NA, k), rows = m_shared_rows(R, k);
                    CHK(urows[ng + k] == rows && rows >= 1 && rows <= MAXR, "R=%d shared unit %d rows %d / urows %d", R, k, rows, urows[ng + k]);
                    for (int row = 0; row < rows; ++row)
                    {
                        const int token = p0 - NA + row;
                        CHK(token >= 0 && token < R && p0 + row >= NA && p0 + row < NAR, "R=%d shared unit %d row %d token %d p0 %d", R, k, row, token, p0);
                        if (token >= 0 && token < R) shseen[token]++;
                        if (p0 + row >= 0 && p0 + row < NAR) outrow[p0 + row]++;
                    }
                    CHK(gu_out<S>(1, NAR, p0) + (long) rows * S::INTER <= (long) 2 * NAR * S::INTER && (long) dn_row<S>(p0) + (long) rows * S::D <= (long) NAR * S::D, "R=%d shared unit %d output rows", R, k);
                }
                for (int r = 0; r < R; ++r) CHK(shseen[r] == 1, "R=%d shared row %d covered %d times", R, r, shseen[r]);
                for (int q = 0; q < NAR; ++q) CHK(outrow[q] == 1, "R=%d gu/dn row %d written %d times", R, q, outrow[q]);
                // segments: gate/up seg = 2 u + proj, down seg = u, all below NU, rows of every seg known to split_total; cost split total is finite and monotone
                std::vector<int> ur(urows.begin(), urows.begin() + NU);
                const long long W = split_total(ur.data(), NU, 2, DM::TPP);
                for (int G : {1, 2, 40, 80, 160})
                {
                    int prev = 0;
                    for (int b = 0; b <= G; ++b)
                    {
                        const int t = split_tile(ur.data(), NU, 2, DM::TPP, (long long) b * W / G);
                        CHK(t >= prev && t <= NU * 2 * DM::TPP, "R=%d G=%d split_tile %d", R, G, t);
                        if (t < NU * 2 * DM::TPP) CHK((t / DM::TPP) >> 1 < NU, "R=%d seg unit %d >= NU", R, (t / DM::TPP) >> 1);
                        prev = t;
                    }
                    CHK(prev == NU * 2 * DM::TPP, "R=%d G=%d split end", R, G);
                }
            }
    }
}

int main()
{
    prove<QwenShape>("Qwen"); prove<QwenK3S4>("QwenK3S4"); prove<QwenK4S5>("QwenK4S5"); prove<SmallTestShape>("SmallTest"); prove<SmallK4S4>("SmallK4S4"); prove<SmallK3S5>("SmallK3S5");
    printf("checks %lld fails %lld\n", checks, fails);
    return fails ? 1 : 0;
}
