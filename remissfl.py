#!/usr/bin/env python3

import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from typing import Callable

work_dir = '/Users/kliron/Projects/remissfl'
selections_dir = os.path.join(work_dir, 'selections')
figures_dir = os.path.join(work_dir, 'figures')
dump = os.path.join(work_dir, 'rtg_huddinge_2010-2019.csv')

print('Loading data from {}...'.format(dump))

df = pd.read_csv(dump, sep='|', dtype={
    'bestallning_uid': str,
    'bestallningstidpunkt': str,
    'remiss_datum': str,
    'remiss_tid': str,
    'prioritet': str,
    'akut': float,
    'bestalld_från_vårdenhet_id': str,
    'vardenhet_namn': str,
    'undersokningstid': str,
    'undersokning': str,
    'till_sektion': str,
    'lab_kombikakod': str,
    'lab_vardenhet': str,
    'svarstyp': str,
    'svar_mottogs': str
},
                 parse_dates=['bestallningstidpunkt', 'remiss_datum', 'remiss_tid', 'undersokningstid', 'svar_mottogs'])

print('\nLoaded DataFrame with {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

# Some columns have changed names between Carestream RIS and Sectra RIS. Fix them.


def is_acute(row: Series) -> float:
    p = row['prioritet']
    a = row['akut']
    if pd.isnull(p):
        return a
    elif p == 'Normal':
        return 0.0
    else:
        return 1.0


df.akut = df.apply(is_acute, axis=1)

akuta = df[df.akut == 1.0]

_rtg = set(akuta[akuta.undersokning.str.contains('rtg', case=False) &
                 ~akuta.undersokning.str.contains('svälj|esofagus|hypofarynx', case=False)]
           .undersokning
           .unique()
           .tolist())

_dt = set(akuta[akuta.undersokning.str.contains('DT|CT', case=False) &
                akuta.undersokning.str.contains('biopsi', case=False)]
          .undersokning
          .unique()
          .tolist())

_glys = set(akuta[akuta.undersokning.str.contains('G-lys|Glys', case=False)]
            .undersokning
            .unique()
            .tolist())

ul = set(akuta[(akuta.undersokning.str.contains(
    r'ul\s|ulj|ultraljud|utraljud|utrajlud|ultrajlud|u-ljud|biopsi|PTC|PD-kateter|Urogenital us|'
    r'UL-|UL/flebografi|Flebografi/ul|Ul/|ulled|duplex|'
    r'ulttraljud|ultaljud|ujl|ultrljud|Ultraljusleds|ultrajudsled|Ul-ledd|ultaljud|Ultrljud|u-ljud|UL:s',
    case=False)) &
               (~akuta.undersokning.isin(_rtg.union(_dt, _glys)))]
         .undersokning
         .unique()
         .tolist())

mr = set(akuta[(akuta.undersokning.str.contains(r'MR\s|MR-|MRT|MRI|magnet', case=False))]
         .undersokning
         .unique()
         .tolist())

nm = set(akuta[(akuta.undersokning.str.contains(
    r'\sPET\s|^PET\s|PET-CT|PETCT|scint|skint|scinitigrafi|scinrigrafi', case=False))]
         .undersokning
         .unique()
         .tolist())

dt = set(akuta[akuta.undersokning.str.contains(
    r'DT\s|CT\s|CT-|DTskalle|CTesofagus|CTurografi|CTansikt|Datortomografi|DT-|halskärl|Passage|Pasagertg|'
    r'Trauma Ct|DTthorax|Käkleder|Öra|Urinvägsöversikt|Colon|Bäckenmätning|Pssageröntgen|DThö|DTvä|Passag-rtg|'
    r'CTlever|CTbuk|DTbuk|CTflebografi',
    case=False
    ) &
        (~akuta.undersokning.isin(mr.union(ul, nm)))]
         .undersokning
         .unique()
         .tolist())

angio = set(akuta[(akuta.undersokning.str.contains(
    r'Angio|TIPS|intervention|stomi|Cava|Interv.|Kärlkateter justering|Dialyskateter|CVP|[CK]oronar\s?angiografi|ERC|'
    'coronarangiografi', case=False)) &
                  (~akuta.undersokning.isin(dt.union(mr, nm, ul, _glys)))]
            .undersokning
            .unique()
            .tolist())


granskning = set(akuta[akuta.undersokning.str.contains('granskning|konf', case=False)]
                 .undersokning.unique()
                 .tolist())

_glys_excl = set(akuta[akuta.undersokning.str.contains('flebografi|phlebografi|mammografi', case=False)].undersokning
                 .unique()
                 .tolist())

glys = set(akuta[(akuta.undersokning.str.contains(
    r'G-lys|Glys|grafi|sväljrtg|Suprapubis|Tunntarm|Sväljningsmotorik|Tarm|invagination|Magsäck|MUC|Larynx|Hypofarynx|'
    r'jejunostomi|Genomlysning|Esofagus|Ventrikelsond|pH-sond nedläggning|Främmande|Kontraströntgen PEG|esofagus'
    , case=False
)) &
          (~akuta.undersokning.isin(dt.union(mr, nm, ul, angio, _rtg, _glys_excl, granskning)))]
           .undersokning
           .unique()
           .tolist())

exclude = '|'.join(['Utlån', 'EKG', 'Sekundärremiss', 'Remissgranskning', 'Pericardpunktion',
                    'Pacemakerinläggning', 'Hjärtkateterisering', 'Gastroskopi', 'Gastrointestinal transittid',
                    'Ekg vila', 'Duodenoskopi',' Dialyskateter', 'CVP inläggning', 'CR - Demonstration',
                    'Bukaortaaneurysm beh', 'Bildlagring', 'Administrativ tjänst på Fysiologen', 'Kärl artär',
                    'Ledig rad', 'Narkosbokning', 'Endoskopi', 'Endoskopiskt ultraljud',
                    'ERCP', 'Efterbearbetning', 'Bildtjänst', 'Provligga/Provåka i modalitet', 'Babygram',
                    'gastroscopi', 'gastroskopi', 'Transesofageal EKO', 'mammografi'])

rtg = set(akuta[~(akuta.undersokning.isin(dt.union(mr, angio, ul, nm, glys, granskning)) |
                  akuta.undersokning.str.contains(exclude))]
          .undersokning
          .unique()
          .tolist())

unclassed = set(akuta[~akuta.undersokning.isin(dt.union(mr, angio, ul, nm, glys, granskning, rtg))]
                .undersokning
                .unique()
                .tolist())


def save_selection_strings(strings: set, name: str):
    with open(os.path.join(selections_dir, name + '.txt'), 'w') as w:
        for v in strings:
            w.write(v + '\n')


for s, n in zip([ul, nm, mr, dt, angio, glys, rtg], ['ul', 'nm', 'mr', 'dt', 'angio', 'glys', 'rtg']):
    save_selection_strings(s, n)


print('\nAdding modality and time interval category to rows...')


def get_modality(row: Series) -> str:
    if row.undersokning in dt:
        return 'DT'
    elif row.undersokning in mr:
        return 'MR'
    elif row.undersokning in nm:
        return 'NM'
    elif row.undersokning in angio:
        return 'Angio'
    elif row.undersokning in ul:
        return 'Ulj'
    elif row.undersokning in glys:
        return 'Glys'
    elif row.undersokning in rtg:
        return 'Rtg'
    else:
        return 'Other'


time1 = datetime.time(7, 30, 00)
time2 = datetime.time(12, 00, 00)
time3 = datetime.time(16, 00, 00)
time4 = datetime.time(20, 00, 00)
time5 = datetime.time(00, 00, 00)


def get_interval_fn_for(col_name: str) -> Callable:
    def get_interval(row: Series) -> int:
        c = row[col_name]
        if not pd.notna(c):
            return np.NaN

        t = c.time()
        if time1 <= t < time2:
            return 1
        elif time2 <= t < time3:
            return 2
        elif time3 <= t < time4:
            return 3
        elif t >= time4:
            return 4
        elif time5 <= t < time1:
            return 5
    return get_interval


def get_year(row: Series) -> int:
    d = row['svar_mottogs'].to_pydatetime()
    return d.year


# Add modality and interval (skapad, svarad) as separate columns so that we are able to group and count them.
get_interval_skickad = get_interval_fn_for('bestallningstidpunkt')
get_interval_svarad = get_interval_fn_for('svar_mottogs')


print('\nCalculating time intervals and deltas. This is going to take a while...')

delta_minutes = (akuta.svar_mottogs - akuta.bestallningstidpunkt).apply(lambda m: m.total_seconds() / 60.0)

akuta = akuta.assign(modalitet=akuta.apply(get_modality, axis=1),
                     interval_skapad=akuta.apply(get_interval_skickad, axis=1),
                     interval_svarad=akuta.apply(get_interval_svarad, axis=1),
                     delta_minutes=delta_minutes,
                     year=akuta.apply(get_year, axis=1))


# Group by year, modality, interval_svarad
gb_alla_svarade = akuta.groupby(['year', 'interval_svarad'])
gb_alla_skapade = akuta.groupby(['year', 'interval_skapad'])
gb_svarade = akuta.groupby(['year', 'modalitet', 'interval_svarad'])
gb_skapade = akuta.groupby(['year', 'modalitet', 'interval_skapad'])

# reset_index call pulls the aggregated columns from the multiindex into a normal dataframe
counts_alla_svarade = gb_alla_svarade.size().reset_index()
counts_alla_svarade.rename(columns={0: 'antal'}, inplace=True)
counts_alla_skapade = gb_alla_skapade.size().reset_index()
counts_alla_skapade.rename(columns={0: 'antal'}, inplace=True)
counts_svarade = gb_svarade.size().reset_index()
counts_svarade.rename(columns={0: 'antal'}, inplace=True)
counts_skapade = gb_skapade.size().reset_index()
counts_skapade.rename(columns={0: 'antal'}, inplace=True)

years = akuta.year.unique().tolist()
years.sort()

# Grand total under jourtid
jour_alla_svarade = counts_alla_svarade[counts_alla_svarade.interval_svarad.isin([3, 4, 5]) &
                                        counts_alla_svarade.year.isin(years)]\
    .groupby(['year'])\
    .sum()\
    .reset_index()

jour_alla_skapade = counts_alla_skapade[counts_alla_skapade.interval_skapad.isin([3, 4, 5]) &
                                        counts_alla_skapade.year.isin(years)] \
    .groupby(['year']) \
    .sum() \
    .reset_index()

sen_jour_alla_svarade = counts_alla_svarade[counts_alla_svarade.interval_svarad.isin([5]) &
                                            counts_alla_svarade.year.isin(years)] \
    .groupby(['year']) \
    .sum() \
    .reset_index()

sen_jour_alla_skapade = counts_alla_skapade[counts_alla_skapade.interval_skapad.isin([5]) &
                                            counts_alla_skapade.year.isin(years)] \
    .groupby(['year']) \
    .sum() \
    .reset_index()


jour_svarade = counts_svarade[counts_svarade.interval_svarad.isin([3, 4, 5]) &
                              counts_svarade.year.isin(years) &
                              counts_svarade.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])]\
    .groupby(['year', 'modalitet'])\
    .sum()\
    .reset_index()

jour_skapade = counts_skapade[counts_skapade.interval_skapad.isin([3, 4, 5]) &
                              counts_skapade.year.isin(years) &
                              counts_skapade.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])]\
    .groupby(['year', 'modalitet'])\
    .sum()\
    .reset_index()

sen_jour_svarade = counts_svarade[counts_svarade.interval_svarad.isin([5]) &
                                  counts_svarade.year.isin(years) &
                                  counts_svarade.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])] \
    .groupby(['year', 'modalitet']) \
    .sum() \
    .reset_index()

sen_jour_skapade = counts_skapade[counts_skapade.interval_skapad.isin([5]) &
                                  counts_skapade.year.isin(years) &
                                  counts_skapade.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])] \
    .groupby(['year', 'modalitet']) \
    .sum() \
    .reset_index()

ej_jour_skapade = counts_skapade[counts_skapade.interval_skapad.isin([1, 2]) &
                                 counts_skapade.year.isin(years) &
                                 counts_skapade.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])]\
    .groupby(['year', 'modalitet'])\
    .sum()\
    .reset_index()


# Plots
def per_year_number_barplot(dfm: DataFrame, title: str):
    ind = np.arange(len(years))
    bar_width = 0.32
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(years)
    ax.set_xlabel('År')
    ax.set_ylabel('Antal')
    ymax = dfm.antal.max()
    ax.set_ylim(0, ymax + ymax*20/100)  # Set y limit higher so that labels don't overlap legend

    # Make the bar plot rectangles
    rects = ax.bar(ind - bar_width/2, dfm['antal'], bar_width)

    # Add counts
    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 200, '%d' % h, ha='center', va='bottom', rotation=90)

    # plt.show()

    # Make room for xlabel otherwise it is clipped when saving to png
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)))


def per_year_modality_number_barplot(dfm: DataFrame, title: str):
    mdt = dfm[dfm.modalitet == 'DT'].antal.values
    mrtg = dfm[dfm.modalitet == 'Rtg'].antal.values
    mulj = dfm[dfm.modalitet == 'Ulj'].antal.values
    mglys = dfm[dfm.modalitet == 'Glys'].antal.values

    ind = np.arange(len(years))
    bar_width = 0.24
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(years)
    ax.set_xlabel('År')
    ax.set_ylabel('Antal')
    ymax = dfm.antal.max()
    ax.set_ylim(0, ymax + ymax*30/100)  # Set y limit higher so that labels don't overlap legend

    # Make the bar plot rectangles
    dt_rects = ax.bar(ind - bar_width/2, mdt, bar_width, label='DT')
    rtg_rects = ax.bar(ind + bar_width/2, mrtg, bar_width, label='Rtg')
    ulj_rects = ax.bar(ind + (bar_width/2)*3, mulj, bar_width, label='Ulj')
    glys_rects = ax.bar(ind + bar_width*2.5, mglys, bar_width, label='Glys')

    ax.legend(ncol=4)

    # Add counts
    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 200, '%d' % h, ha='center', va='bottom', rotation=90)

    # plt.show()

    # Make room for xlabel otherwise it is clipped when saving to png
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)))


# def per_year_timedelta_barplot(dfm: DataFrame, title: str):
#     mdt = dfm[dfm.modalitet == 'DT'].delta_minutes.values
#     mrtg = dfm[dfm.modalitet == 'Rtg'].delta_minutes.values
#     mulj = dfm[dfm.modalitet == 'Ulj'].delta_minutes.values
#     mglys = dfm[dfm.modalitet == 'Glys'].delta_minutes.values
#
#     ind = np.arange(len(years))
#     bar_width = 0.24
#     fig, ax = plt.subplots()
#     ax.set_title(title)
#     ax.set_xticks(ind)
#     ax.set_xticklabels(years)
#     ax.set_xlabel('År')
#     ax.set_ylabel('Minuter')
#     ax.set_ylim(0, dfm.delta_minutes.max() + 6000)  # Set y limit higher so that labels don't overlap legend
#
#     # Make the bar plot rectangles
#     dt_rects = ax.bar(ind - bar_width/2, mdt, bar_width, label='DT')
#     rtg_rects = ax.bar(ind + bar_width/2, mrtg, bar_width, label='Rtg')
#     ulj_rects = ax.bar(ind + (bar_width/2)*3, mulj, bar_width, label='Ulj')
#     glys_rects = ax.bar(ind + bar_width*2.5, mglys, bar_width, label='Glys')
#
#     ax.legend(ncol=4, mode='expand')
#
#     # Add counts
#     for rect in ax.patches:
#         h = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2., h + 200, '%d' % h, ha='center', va='bottom', rotation=90)
#
#     # plt.show()
#
#     # Make room for xlabel otherwise it is clipped when saving to png
#     plt.gcf().subplots_adjust(bottom=0.15)
#
#     plt.savefig('/Users/kliron/Downloads/{}.png'.format(title))


print('\nPlotting results...')

plt.style.use('seaborn')

per_year_number_barplot(jour_alla_skapade, 'Akuta remisser skapade 16.00 - 07.30 (alla modaliteter)')
per_year_number_barplot(jour_alla_svarade, 'Akuta remisser besvarade 16.00 - 07.30 (alla modaliteter)')
per_year_number_barplot(sen_jour_alla_skapade, 'Akuta remisser skapade 00.00 - 07.30 (alla modaliteter)')
per_year_number_barplot(sen_jour_alla_svarade, 'Akuta remisser besvarade 00.00 - 07.30 (alla modaliteter)')
per_year_modality_number_barplot(jour_skapade, 'Akuta remisser skapade 16.00 - 07.30')
per_year_modality_number_barplot(jour_svarade, 'Akuta remisser besvarade 16.00 - 07.30')
per_year_modality_number_barplot(sen_jour_skapade, 'Akuta remisser skapade 00.00 - 07.30')
per_year_modality_number_barplot(sen_jour_svarade, 'Akuta remisser besvarade 00.00 - 07.30')
per_year_modality_number_barplot(ej_jour_skapade, 'Akuta remisser skapade 07.30 - 16.00')
# per_year_timedelta_barplot(akuta, 'Interval mellan remiss skickad och svarad')
