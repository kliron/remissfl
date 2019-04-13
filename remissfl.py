#!/usr/bin/env python3

import pandas as pd
from pandas import DataFrame, Series, Timedelta
import numpy as np
import os
import re
import datetime
import matplotlib.pyplot as plt
from typing import Callable
from collections import Counter


work_dir = './'
selections_dir = os.path.join(work_dir, 'selections')
figures_dir = os.path.join(work_dir, 'figures')
xlsx_dir = os.path.join(work_dir, 'xlsx')
dump = os.path.join(work_dir, 'rtg_huddinge_2010-2019.csv')

print('Läser data från {}...'.format(dump))

_df = pd.read_csv(dump, sep='|', dtype={
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

print('\nLaddat DataFrame med {} rader och {} kolumner.'.format(_df.shape[0], _df.shape[1]))


# 1. Kolumn 'prioritet' har döpts om till 'akut' mellan Carestream och Sectra RIS. Slå ihop.

def is_acute(row: Series) -> float:
    p = row['prioritet']
    a = row['akut']
    if pd.isnull(p):
        return a
    elif p == 'Normal':
        return 0.0
    else:
        return 1.0


_df.akut = _df.apply(is_acute, axis=1)

akuta = _df[_df.akut == 1.0]

# 2. Modalitet finns inte som separat variabel i TC. Vi behöver gissa modalitet baserat på "undersokning" kolumnen
#    vilken är fritext. Det är nästan omöjligt att sortera alla rätt, små fel kommer att finnas men vi behöver veta att
#    viktiga texter (som upprepas i hundratals remisser t.ex) inte hamnar fel.

# Räkna alla unika 'undersokning' värden
undersokning_counts = Counter(akuta.undersokning.to_list())


def percent_covered_by_top_n(n: int = 500) -> float:
    """Return the percent of the total number of 'undersokning' values covered by the n most common"""
    c = undersokning_counts.most_common(n)
    total = akuta.shape[0]
    i = 0
    for u, j in c:
        i += j
    return (i/total) * 100


top_n = 500
print(
    '\nDet finns {} unika värden i "undersokning" kolumn. De {} mest frekventa representerar {:.1f}% av hela summan.'
    .format(len(undersokning_counts), top_n, percent_covered_by_top_n(n=top_n))
)

# 3. Sortera per modalitet

_rtg_sel = r'rtg|skoliosrygg|röntgen|lungröngten|HKA|pulm|Skelettålder|gips|^lungor$|^BÖS\??$|^Buköversikt$|' \
           r'skallfrontal|myelomskelett'
_rtg = set(akuta[akuta.undersokning.str.contains(_rtg_sel, case=False) &
                 ~akuta.undersokning.str.contains('svälj|esofagus|hypofarynx', case=False)]
           .undersokning
           .unique()
           .tolist())

_dt = set(akuta[akuta.undersokning.str.contains(r'[DC]T', case=False) &
                akuta.undersokning.str.contains(r'biopsi', case=False)]
          .undersokning
          .unique()
          .tolist())

_glys_sel = r'G-lys|Glys|genomlysning|svälj|övre.*?passage|passage.*?övre|[eö]sofagus.*?passage|duodenum.*?passage|'\
            r'passage.*?[öe]sofagus|övre buk.*?passage'
_glys = set(akuta[akuta.undersokning.str.contains(_glys_sel, case=False)]
            .undersokning
            .unique()
            .tolist())

ul_sel = r'ul\s|ulj|ultraljud|utraljud|utrajlud|ultrajlud|u-ljud|biopsi|PTC|PD-kateter|Urogenital us|UL-|' \
         r'UL/flebografi|Flebografi/ul|Ul/|ulled|duplex|ulttraljud|ultaljud|ujl|ultrljud|Ultraljusleds|ultrajudsled|' \
         r'Ul-ledd|ultaljud|Ultrljud|u-ljud|UL:s|Ultrajud|Tappning av pleuravätska|Tappning pleuravätska|UL\.|^ul$|' \
         r'Per op UL|lever UL|Pleuradrän|pleuratappning|^UL$|\sUL$|^Ul,|Pleura tappning|Bukultarljud|Uj:|Ultraljug|' \
         r'Uld |U/L |ullj |Ultarljud|Lungpunktion|ULD |ULlever|Ult '
ul = set(akuta[(akuta.undersokning.str.contains(ul_sel, case=False)) &
               (~akuta.undersokning.isin(_rtg.union(_dt, _glys)))]
         .undersokning
         .unique()
         .tolist())

mr_sel = r'MR\s|MR-|MRT|MRI|MRhö|MRvä|MRCP|magnet|\sMR|MRC |MRhjärn|^MR[TI]?$|MRlever|hjärnMR|^MRC'
mr = set(akuta[(akuta.undersokning.str.contains(mr_sel, case=False))]
         .undersokning
         .unique()
         .tolist())

nm_sel = r'\sPET\s|^PET\s|\sPET|-PET|PET/CT|PET-|FDG|PETCT|SPECT|DAT-|DMSA|scint|skint|scinitigrafi|scinrigrafi|' \
         r'scintografi|scinnt|^PET$|Ventrikeltömningstest|MAG-?3|Renogram|scynt|DATSCAN|PET/[DC]T|[DC]Ttorax'
nm = set(akuta[(akuta.undersokning.str.contains(nm_sel, case=False))]
         .undersokning
         .unique()
         .tolist())

dt_sel = r'DT\s|CT\s|[CD]Tai|[DC]Ttrauma|[CD]Tskalle|[DC]Tesofagus|[CD]Turografi|[CD]Tansikt|Datortomografi|[CD]T-|' \
         r'halskärl|Passage|Pasagertg|Trauma Ct|DTthorax|Käkleder|Öra|Urinvägsöversikt|Colon|Bäckenmätning|' \
         r'Pssageröntgen|DThö|DTvä|Passag-rtg|[DC]Tlever|[DC]Tbuk|[DC]Tflebografi|[CD]TBÖS|[DC]Tstenöversikt|[CD]T\.|' \
         r'Buk [CD]T|[CD]Thals|spiral-?[DC]T|HRCT|lungemboli|[CD]Thjärna|D.T |Trombolys|HR CT|[DC]T_|' \
         r'Volymmätning av bukfett och muskler|-[CD]T|\s[DC]T|^[CD]T$|^esophagus$|[CD]Tthorax|hCRT|[DC]/T |CBCT'
dt = set(akuta[akuta.undersokning.str.contains(dt_sel, case=False) &
               (~akuta.undersokning.isin(mr.union(ul, nm, _glys)))]
         .undersokning
         .unique()
         .tolist())

angio_sel = r'Angio|TIPS|intervention|stomi|Cava|Interv.|Kärlkateter justering|Dialyskateter|CVP|' \
            r'[CK]oronar\s?angiografi|ERC|coronarangiografi|PCI'
angio = set(akuta[(akuta.undersokning.str.contains(angio_sel, case=False)) &
                  (~akuta.undersokning.isin(dt.union(mr, nm, ul, _glys)))]
            .undersokning
            .unique()
            .tolist())

granskning_sel = r'granskning|konf|demo|rond'
granskning = set(akuta[akuta.undersokning.str.contains(granskning_sel, case=False)]
                 .undersokning.unique()
                 .tolist())

_glys_excl_sel = r'flebo|phlebografi|mammografi'
_glys_excl = set(akuta[akuta.undersokning.str.contains(_glys_excl_sel, case=False)].undersokning
                 .unique()
                 .tolist())

glys_sel = r'G-lys|Glys|grafi|sväljrtg|Suprapubis|Tunntarm|Sväljningsmotorik|Tarm|invagination|Magsäck|MUC|Larynx|' \
           r'Hypofarynx|jejunostomi|Genomlysning|Esofagus|Ventrikelsond|pH-sond nedläggning|Främmande|' \
           r'Kontraströntgen PEG|esofagus|Kontroll nefrostomi|hypopharynx/oesophagus|Genomylsning|Fluoroskopi|' \
           r'Esofagus-?passage|sväljningspassage|passage'
glys = set(akuta[(akuta.undersokning.str.contains(glys_sel, case=False)) &
           (~akuta.undersokning.isin(dt.union(mr, nm, ul, angio, _rtg, _glys_excl, granskning)))]
           .undersokning
           .unique()
           .tolist())

exclude = '|'.join(['Utlån', 'EKG', 'Sekundärremiss', 'Remissgranskning', 'Pericardpunktion',
                    'Pacemakerinläggning', 'Hjärtkateterisering', 'Gastroskopi', 'Gastrointestinal transittid',
                    'Ekg vila', 'Duodenoskopi',' Dialyskateter', 'CVP inläggning', 'CR - Demonstration',
                    'Bukaortaaneurysm beh', 'Bildlagring', 'Administrativ tjänst på Fysiologen', 'Kärl artär',
                    'Ledig rad', 'Narkosbokning', 'Endoskopi', 'Endoskopiskt ultraljud', '[CK]oloskopi'
                    'ERCP', 'Efterbearbetning', 'Bildtjänst', 'Provligga/Provåka i modalitet', 'Babygram',
                    'gastroscopi', 'gastroskopi', 'Transesofageal EKO', 'mammografi', 'Koppling av bilde', 'länkning',
                    'SCAPIS', 'Inscanning', 'Anpassningsremiss'])

rtg = set(akuta[~(akuta.undersokning.isin(dt.union(mr, angio, ul, nm, glys, granskning)) |
                  akuta.undersokning.str.contains(exclude))]
          .undersokning
          .unique()
          .tolist())

unclassed = set(akuta[~akuta.undersokning.isin(dt.union(mr, angio, ul, nm, glys, granskning, rtg))]
                .undersokning
                .unique()
                .tolist())


# Har vi missat viktiga strängar? Kör våra selektorer regexes mot de viktigaste N

def matches_any_selector(s: str) -> bool:
    for sel in [exclude, glys_sel, angio_sel, ul_sel, dt_sel, mr_sel, nm_sel, granskning_sel, _rtg_sel, _glys_excl_sel]:
        if (not pd.isnull(s)) and (re.search(sel, s, re.IGNORECASE)):
            return True
    return False


most_common = 10000
print('\nTestar om vi har missat frekventa strängar i de {} vanligaste (dessa täcker {:.2f}% av alla värden).'
      .format(most_common, percent_covered_by_top_n(n=most_common)))

important = undersokning_counts.most_common(most_common)

missed = []
for _s, _c in important:
    if not matches_any_selector(_s):
        missed.append((_s, _c))

n_missed = 0
for _m in missed:
    n_missed += _m[1]

print('\nTotalt missade: {}/{} (i {} remisser)\nViktigast: "{}" med {} counts'
      .format(len(missed), most_common, n_missed, missed[0][0], missed[0][1]))


def save_selection_strings(strings: set, name: str):
    with open(os.path.join(selections_dir, name + '.txt'), 'w') as w:
        for v in strings:
            w.write(v + '\n')


for _s, _n in zip([ul, nm, mr, dt, angio, glys, rtg], ['ul', 'nm', 'mr', 'dt', 'angio', 'glys', 'rtg']):
    save_selection_strings(_s, _n)


print('\nLägger till modalitet och tidsintervall.')


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
    elif row.undersokning in granskning:
        return 'Granskning'
    else:
        return 'Annat'


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
    return row['svar_mottogs'].to_pydatetime().year


def get_month(row: Series) -> int:
    return row['svar_mottogs'].to_pydatetime().month  # börjar från 1


def get_weekday(row: Series) -> int:
    return row['svar_mottogs'].to_pydatetime().weekday()  # börjar från 0 (0 = måndag)


def get_day(row: Series) -> int:
    return row['svar_mottogs'].to_pydatetime().day


# Add modality and interval (skapad, svarad) as separate columns so that we are able to group and count them.
get_interval_skickad = get_interval_fn_for('bestallningstidpunkt')
get_interval_svarad = get_interval_fn_for('svar_mottogs')


print('\nBeräknar tidsintervaller och deltas...')

delta_hours = (akuta.svar_mottogs - akuta.bestallningstidpunkt).apply(lambda m: m.total_seconds() / 3600.0)

akuta = akuta.assign(modalitet=akuta.apply(get_modality, axis=1),
                     interval_skapad=akuta.apply(get_interval_skickad, axis=1),
                     interval_svarad=akuta.apply(get_interval_svarad, axis=1),
                     delta_t=delta_hours,
                     year=akuta.apply(get_year, axis=1),
                     month=akuta.apply(get_month, axis=1),
                     day=akuta.apply(get_day, axis=1),
                     weekday=akuta.apply(get_weekday, axis=1))


# Group by year, modality, interval_svarad
alla_svarade = akuta.groupby(['year', 'interval_svarad'])
alla_skapade = akuta.groupby(['year', 'interval_skapad'])
svarade = akuta.groupby(['year', 'modalitet', 'interval_svarad'])
skapade = akuta.groupby(['year', 'modalitet', 'interval_skapad'])

# reset_index call pulls the aggregated columns from the multiindex into a normal dataframe
counts_alla_svarade = alla_svarade.size().reset_index(name='antal')
counts_alla_skapade = alla_skapade.size().reset_index(name='antal')
counts_svarade = svarade.size().reset_index(name='antal')
counts_skapade = skapade.size().reset_index(name='antal')

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


##################################################
# Vilka månader och veckodagar er mer belastade? #
##################################################


def date_to_str(row: Series) -> str:
    d = row['svar_mottogs'].to_pydatetime()
    return '{0}-{1:02d}-{2:02d}'.format(d.year, d.month, d.day)


days = ('Mån', 'Tis', 'Ons', 'Tor', 'Fre', 'Lör', 'Sön')
months = ('Jan', 'Feb', 'Mar', 'Apr', 'Maj', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec')

_akuta_dagtid = akuta[akuta.interval_svarad.isin([1, 2])]
_akuta_jourtid = akuta[akuta.interval_svarad.isin([3, 4, 5])]

_dagtid_by_month_and_day = _akuta_dagtid.groupby(['year', 'month', 'day']).size().reset_index(name='antal')
_jourtid_by_month_and_day = _akuta_jourtid.groupby(['year', 'month', 'day']).size().reset_index(name='antal')

busiest_month_dagtid = _dagtid_by_month_and_day.loc[_dagtid_by_month_and_day.antal == _dagtid_by_month_and_day.antal.max(),
                                 ['year', 'month', 'antal']]\
    .values.tolist()[0]

print('\nMest belastad månad dagtid var {} {} med {} akuta remisser besvarade 07:30 - 16:00'
      .format(months[int(busiest_month_dagtid[1])], busiest_month_dagtid[0], busiest_month_dagtid[2]))

busiest_month_jourtid = _jourtid_by_month_and_day.loc[_jourtid_by_month_and_day.antal == _jourtid_by_month_and_day.antal.max(),
                                                      ['year', 'month', 'antal']] \
    .values.tolist()[0]

print('\nMest belastad månad jourtid var {} {} med {} akuta remisser besvarade 16:00 - 07:30'
      .format(months[int(busiest_month_jourtid[1])], busiest_month_jourtid[0], busiest_month_jourtid[2]))


busiest_day_dagtid = _dagtid_by_month_and_day.loc[_dagtid_by_month_and_day.antal == _dagtid_by_month_and_day.antal.max(),
                                                  ['year', 'month', 'day', 'antal']] \
    .values.tolist()[0]

_weekday1 = days[datetime.datetime(busiest_day_dagtid[0],
                                   busiest_day_dagtid[1],
                                   busiest_day_dagtid[2]).weekday()]

print('\nMest belastad dag var {} {}/{} {} med {} akuta remisser besvarade 07:30 - 16:00'
      .format(_weekday1, busiest_day_dagtid[2], busiest_day_dagtid[1], busiest_day_dagtid[0], busiest_day_dagtid[3]))

busiest_day_jourtid = _jourtid_by_month_and_day.loc[_jourtid_by_month_and_day.antal == _jourtid_by_month_and_day.antal.max(),
                                                  ['year', 'month', 'day', 'antal']] \
    .values.tolist()[0]

_weekday2 = days[datetime.datetime(busiest_day_jourtid[0],
                                   busiest_day_jourtid[1],
                                   busiest_day_jourtid[2]).weekday()]

print('\nMest belastad jour var {} {}/{} {} med {} akuta remisser besvarade 16:00 - 07:30'
      .format(_weekday1, busiest_day_jourtid[2], busiest_day_jourtid[1], busiest_day_jourtid[0], busiest_day_jourtid[3]))

_dagtid_by_month = _akuta_dagtid.groupby(['year', 'month']).size().reset_index(name='antal')
_jourtid_by_month = _akuta_jourtid.groupby(['year', 'month']).size().reset_index(name='antal')
_dagtid_by_weekday = _akuta_dagtid.groupby(['weekday', 'year']).size().reset_index(name='antal')
_jourtid_by_weekday = _akuta_jourtid.groupby(['weekday', 'year']).size().reset_index(name='antal')

_dagtid_by_month_and_weekday = _akuta_dagtid.groupby(['year', 'month', 'weekday']).size().reset_index(name='antal')
_jourtid_by_month_and_weekday = _akuta_jourtid.groupby(['year', 'month', 'weekday']).size().reset_index(name='antal')

_dagtid_by_month_and_weekday.to_excel(os.path.join(xlsx_dir, 'dagtid_per_månad_och_veckodag.xlsx'), index=False)
_jourtid_by_month_and_weekday.to_excel(os.path.join(xlsx_dir, 'jourtid_per_månad_och_veckodag.xlsx'), index=False)


#############################################################
# Olika joursystem. Datum nya systemet infördes: 2019-02-11 #
#############################################################

# inrem ska ringa för us efter kl 00:00 = system 0
# inrem ska inte ringa för us efter kl 00:00 = system 1
olika_system = akuta[akuta.year.isin([2018, 2019]) &
                     (akuta.interval_skapad == 5) &
                     akuta.modalitet.isin(['DT', 'Rtg', 'Glys', 'Ulj'])]

olika_system = olika_system.assign(system=np.where(olika_system.bestallningstidpunkt < datetime.datetime(2019, 2, 11), 0, 1),
                                   datum=olika_system.apply(date_to_str, axis=1))

system_counts = olika_system.groupby(['system', 'datum', 'modalitet']).size().reset_index(name='antal')
system_means = system_counts.groupby(['system', 'modalitet']).mean().add_prefix('medel_')

_w1 = pd.ExcelWriter(os.path.join(xlsx_dir, 'joursystem_nya_vs_gamla.xlsx'))
system_counts.groupby(['system', 'modalitet']).describe().to_excel(_w1, startcol=0, startrow=3)
ws1 = _w1.sheets['Sheet1']
ws1.write_string(0, 0, 'Genomsnitt antal remisser i nya systemet (fr.o.m 2019-02-11) vs gamla')
_w1.save()


# Remove extreme (non-acute) records: keep only records where svar_mottogs is within 24h of bestallningtidpunkt
_dd = _akuta_dagtid.loc[
    (_akuta_dagtid.svar_mottogs - _akuta_dagtid.bestallningstidpunkt) <= Timedelta('1 days 00:00:00')]

deltas_dagtid = _dd[['year', 'delta_t']]

_dj = _akuta_jourtid.loc[
    (_akuta_jourtid.svar_mottogs - _akuta_jourtid.bestallningstidpunkt) <= Timedelta('1 days 00:00:00')]

deltas_jour = _dj[['year', 'delta_t']]

_w2 = pd.ExcelWriter(os.path.join(xlsx_dir, 'tid_deltas_dagtid.xlsx'))
_dd[['year', 'month', 'weekday', 'delta_t']].groupby(['year', 'month', 'weekday']).describe()\
    .to_excel(_w2, startcol=0, startrow=3)
ws2 = _w2.sheets['Sheet1']
ws2.write_string(0, 0, 'Tid (i timmar) det tar för att svara på akuta remisser dagtid. '
                       'Endast remisser besvarade inom 24 timmar räknas.')
_w2.save()

_w3 = pd.ExcelWriter(os.path.join(xlsx_dir, 'tid_deltas_jourtid.xlsx'))
_dj[['year', 'month', 'weekday', 'delta_t']].groupby(['year', 'month', 'weekday']).describe() \
    .to_excel(_w3, startcol=0, startrow=3)
ws3 = _w3.sheets['Sheet1']
ws3.write_string(0, 0, 'Tid (i timmar) det tar för att svara på akuta remisser jourtid. '
                       'Endast remisser besvarade inom 24 timmar räknas.')
_w3.save()


#########
# Plots #
#########


def per_year_counts_barplot(dfm: DataFrame, title: str):
    ind = np.arange(len(years))
    bar_width = 0.32
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticks(ind)
    ax.set_xticklabels(years)
    ax.set_xlabel('År')
    ax.set_ylabel('Antal')
    ymax = dfm.antal.max()
    ax.set_ylim(0, ymax + ymax*0.2)  # Set y limit higher so that labels don't overlap legend

    # Make the bar plot rectangles
    rects = ax.bar(ind - bar_width/2, dfm['antal'], bar_width)

    # Add counts
    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 200, '%d' % h, ha='center', va='bottom', rotation=90)

    # Make room for xlabel otherwise it is clipped when saving to png
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)), dpi=600)


def per_year_modality_counts_barplot(dfm: DataFrame, title: str):
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
    ax.set_ylim(0, ymax + ymax*0.3)  # Set y limit higher so that labels don't overlap legend

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

    # Make room for xlabel otherwise it is clipped when saving to png
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)), dpi=600)


def counts_per_month_boxplot(df: DataFrame, title: str):
    values = [df[df['month'] == m]['antal'].values for m in range(1, 13)]
    fig, ax = plt.subplots()
    plt.boxplot(values)
    ax.set_xticklabels(months)
    ax.set_xlabel('Månad')
    ax.set_ylabel('Antal')
    ax.set_title(title)
    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)), dpi=600)


def counts_per_weekday_boxplot(df: DataFrame, title: str):
    values = [df[df['weekday'] == m]['antal'].values for m in range(0, 7)]
    fig, ax = plt.subplots()
    plt.boxplot(values)
    ax.set_xticklabels(days)
    ax.set_xlabel('Veckodag')
    ax.set_ylabel('Antal')
    ax.set_title(title)
    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)), dpi=600)


def timedelta_boxplot(df: DataFrame, title: str):
    values = [df[df['year'] == y]['delta_t'].values for y in years]
    fig, ax = plt.subplots()
    plt.boxplot(values)
    ax.set_xticklabels(years)
    ax.set_xlabel('År')
    ax.set_ylabel('Timmar')
    ax.set_title(title)
    plt.savefig(os.path.join(figures_dir, '{}.png'.format(title)), dpi=600)


print('\nPlotting results...')

plt.style.use('seaborn')

per_year_counts_barplot(jour_alla_skapade, 'Akuta remisser skapade 16.00 - 07.30 (alla modaliteter)')
per_year_counts_barplot(jour_alla_svarade, 'Akuta remisser besvarade 16.00 - 07.30 (alla modaliteter)')
per_year_counts_barplot(sen_jour_alla_skapade, 'Akuta remisser skapade 00.00 - 07.30 (alla modaliteter)')
per_year_counts_barplot(sen_jour_alla_svarade, 'Akuta remisser besvarade 00.00 - 07.30 (alla modaliteter)')
per_year_modality_counts_barplot(jour_skapade, 'Akuta remisser skapade 16.00 - 07.30')
per_year_modality_counts_barplot(jour_svarade, 'Akuta remisser besvarade 16.00 - 07.30')
per_year_modality_counts_barplot(sen_jour_skapade, 'Akuta remisser skapade 00.00 - 07.30')
per_year_modality_counts_barplot(sen_jour_svarade, 'Akuta remisser besvarade 00.00 - 07.30')
per_year_modality_counts_barplot(ej_jour_skapade, 'Akuta remisser skapade 07.30 - 16.00')
counts_per_month_boxplot(_dagtid_by_month, title='Akuta besvarade remisser per månad, dagtid')
counts_per_month_boxplot(_jourtid_by_month, title='Akuta besvarade remisser per månad, jour')
counts_per_weekday_boxplot(_dagtid_by_weekday, title='Akuta besvarade remisser per veckodag, dagtid')
counts_per_weekday_boxplot(_jourtid_by_weekday, title='Akuta besvarade remisser per veckodag, jour')
timedelta_boxplot(deltas_dagtid, 'Tidsinterval, akuta remisser besvarade inom 24t, dagtid')
timedelta_boxplot(deltas_jour, 'Tidsinterval, akuta remisser besvarade inom 24t, jour')


