import argparse
import pandas as pd

CLASSES = ['main', 'secondary', 'tertiary', 'local']


def agp(l0, l1, years):
    return ((l1 / l0) ** (1.0 / years) - 1.0) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lengths', required=True)
    ap.add_argument('--base_year', type=int, default=2015)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    T = pd.read_csv(a.lengths)
    T['total'] = T[CLASSES].sum(axis=1)
    rows = []
    for (c, s), g in T[T.year > a.base_year].groupby(['country', 'ssp']):
        base = T[(T.country == c) & (T.year == a.base_year)].iloc[0]
        g = g.sort_values('year')
        prev, prev_year = base, a.base_year
        for _, r in g.iterrows():
            for k in CLASSES + ['total']:
                rows.append(dict(country=c, ssp=s, year=int(r.year), road_class=k, length_km=r[k],
                                 agp_since_base=round(agp(base[k], r[k], r.year - a.base_year), 2),
                                 agp_previous_period=round(agp(prev[k], r[k], r.year - prev_year), 2),
                                 previous_year=prev_year,
                                 share_pct=round(100.0 * r[k] / r['total'], 1)))
            prev, prev_year = r, int(r.year)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
