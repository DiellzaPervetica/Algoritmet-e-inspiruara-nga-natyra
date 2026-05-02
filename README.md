<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/University_of_Prishtina_logo.svg/1200px-University_of_Prishtina_logo.svg.png" width="120" alt="University Logo" />
    </td>
    <td valign="top">
      <p>Universiteti i Prishtinës</p>
      <p>Fakulteti i Inxhinierisë Elektrike dhe Kompjuterike</p>
      <p>Inxhinieri Kompjuterike dhe Softuerike - Programi Master</p>
      <p>Profesor: Prof. Dr. Kadri Sylejmani</p>
      <p>Asistent: MSc. Labeat Arbneshi</p>
    </td>
  </tr>
</table>

---
## Përshkrimi i Projektit: Optimizimi i Orarit Televiziv

Ky projekt adreson **Problemin e Planifikimit Televiziv për Hapësira Publike** (TV Channel Scheduling Optimization for Public Spaces) në kuadër të lëndës **Algoritmet e Inspiruara nga Natyra**. Objektivi primar është përzgjedhja dhe planifikimi optimal i një nënbashkësie të programeve televizive në kanale të shumta, me qëllim maksimizimin e pikëve totale të shikueshmërisë.

**Kufizimet dhe Qëllimet Kryesore:**

Përveç kufizimeve bazë kohore, problemi përfshin rregulla specifike për të siguruar një përvojë cilësore shikimi:

*   **Time Window Constraint:** Programet duhet të planifikohen strikt brenda intervalit kohor global të përcaktuar (Hapja dhe Mbyllja).
*   **No Overlap Constraint:** Ndalohet rreptësisht mbivendosja kohore e programeve në të njëjtin kanal.
*   **Minimum Duration:** Programet duhet të kenë një kohëzgjatje minimale për t'u konsideruar të vlefshme.
*   **Genre Repetition:** Për të siguruar shumëllojshmëri, ka një kufizim në numrin e programeve të njëpasnjëshme të të njëjtit zhanër.
*   **Priority Blocks:** Blloqe kohore specifike ku vetëm kanale të caktuara kanë prioritet ose lejohen të transmetojnë.
*   **Time Preferences:** Bonuse pikësh për transmetimin e zhanreve të caktuara në orare të preferuara.
*   **Optimization Goal:** Maksimizimi i funksionit objektiv, duke balancuar pikët e programeve me penalitetet e mundshme.

## Zgjidhjet Fillestare

Para algoritmit gjenetik jane implementuar dy metoda per gjenerimin e zgjidhjeve fillestare:

```text
1. Beam Search Scheduler
2. Branch and Bound Scheduler
```

Keto metoda sherbejne si baze krahasimi dhe si menyre per te krijuar zgjidhje fillestare te mira.

## Beam Search Scheduler

Beam Search Scheduler eshte nje metode heuristike qe mban disa zgjidhje te pjesshme me te mira ne cdo hap, ne vend qe te ndjeke vetem nje zgjidhje te vetme.

Ideja kryesore:

- ndertohet orari hap pas hapi;
- ne cdo hap ruhen disa kandidatet me te mire;
- perdoret nje vleresim heuristik per te zgjedhur drejtimet me premtuese;
- shmanget nje pjese e gabimeve qe mund t'i beje nje algoritm greedy.

Parametrat kryesore te Beam Search:

| Parameter | Value | Pershkrim |
|---|---:|---|
| beam_width | 100 | Numri i zgjidhjeve te pjesshme qe ruhen ne cdo hap |
| lookahead_limit | 4 | Sa hapa perpara analizohet ndikimi i nje vendimi |

Beam Search eshte perdorur si metode fillestare dhe si pike krahasimi, por per algoritmin gjenetik kemi vendosur te perdorim Branch and Bound si zgjidhje iniciale, sepse output-et e tij kane qene me te pershtatshme per warm start.

## Branch and Bound Scheduler

Branch and Bound Scheduler eksploron hapesiren e zgjidhjeve duke krijuar degezime te mundshme dhe duke larguar drejtimet qe nuk duken premtuese.
Kjo metode eshte perdorur per te krijuar zgjidhje fillestare te forta, te cilat me pas sherbejne si pike nisjeje per algoritmin gjenetik.

Ideja kryesore:

- ndertohen zgjidhje te pjesshme duke ecur neper kohe;
- ne cdo pike vendimi gjenerohen programet kandidate qe mund te shtohen ne orar;
- per cdo kandidat llogaritet score-i duke perfshire score-in baze, bonuset, penalitetet dhe kufizimet;
- per cdo degezim vleresohet potenciali maksimal qe mund te arrihet me pjesen e mbetur te dites;
- degezimet qe nuk mund te japin rezultat me te mire se zgjidhja aktuale me e mire largohen;
- ruhet zgjidhja me score me te larte qe gjendet brenda kohes se lejuar.

Komponentet kryesore te implementimit:

| Komponenti | Qellimi |
|---|---|
| Preprocessing | Rendit programet sipas kanaleve dhe koheve, pergatit priority blocks dhe vlerat optimiste per pruning |
| Randomized warm starts | Krijon disa zgjidhje fillestare te shpejta per te pasur nje incumbent te mire para kerkimit kryesor |
| Candidate generation | Gjen programet qe mund te vendosen ne kohen aktuale pa shkelur kufizimet |
| Upper bound | Vlereson kufirin maksimal teorik qe mund te arrihet nga nje gjendje e pjesshme |
| Pruning | Ndal eksplorimin e degezimeve qe nuk mund ta kalojne score-in me te mire aktual |
| DFS search | Eksploron degezimet me stack dhe perditeson zgjidhjen me te mire |

Arsyeja pse kjo metode eshte e pershtatshme per warm start eshte se zakonisht prodhon zgjidhje valide dhe me score te mire. Algoritmi gjenetik nuk fillon nga nje orar plotesisht random, por nga nje zgjidhje qe tashme respekton kufizimet kryesore dhe ka score te larte.

Output-et e Branch and Bound ruhen ne:

```text
data/output/branchandboundscheduler
```

Ne algoritmin gjenetik, keto output-e jane perdorur si **zgjidhje iniciale te gatshme**.

E rendesishme:

```text
Algoritmi gjenetik nuk e gjeneron prape zgjidhjen fillestare me Branch and Bound.
Ai e lexon zgjidhjen ekzistuese nga output-et e ruajtura te Branch and Bound.
```

Kjo qasje u zgjodh per dy arsye:

- kursen kohe gjate ekzekutimit te algoritmit gjenetik;
- e ben eksperimentin me te kontrollueshem, sepse cdo run niset nga e njejta zgjidhje iniciale per ate instance.

Total score i zgjidhjeve fillestare Branch and Bound per 17 instancat ishte:

```text
136573
```

## Algoritmi Gjenetik

Algoritmi gjenetik perdoret per te permiresuar zgjidhjet fillestare te marra nga Branch and Bound.

Ne kete projekt, nje individ ne popullate perfaqeson nje orar te plote ose te pjesshem:

```text
individual = liste e programeve te planifikuara
```

Popullata fillestare krijohet duke marre zgjidhjen nga Branch and Bound dhe duke krijuar variante te reja me mutacion dhe repair.

### Struktura e Algoritmit

Algoritmi funksionon keshtu:

1. Lexohet instance nga `data/input`.
2. Lexohet output-i perkates nga `data/output/branchandboundscheduler`.
3. Krijohet popullata fillestare nga zgjidhja Branch and Bound.
4. Llogaritet fitness per secilin individ.
5. Zgjidhen prinderit me tournament selection.
6. Krijohen femije me crossover.
7. Aplikohet mutacioni.
8. Aplikohet repair per te ruajtur validitetin.
9. Popullata perditesohet duke ruajtur elitat dhe femijet me te mire.
10. Ruhet gjithmone zgjidhja me e mire e gjetur deri ne ate moment.

### Funksionet Kryesore ne Implementim

Implementimi eshte ndare ne funksione te vogla, ku secili funksion ka nje rol te qarte:

| Funksioni | Roli |
|---|---|
| `_base_schedule` | Merr zgjidhjen fillestare nga Branch and Bound dhe e filtron qe te jete valide |
| `_initial_population` | Krijon popullaten fillestare duke ruajtur bazen dhe duke krijuar variante me mutacion |
| `_fitness` | Llogarit score-in e nje individi pas filtrimit valid |
| `_select_population` | Rendit popullaten, ruan elitat dhe krijon mating pool |
| `_pick_parent` | Zgjedh nje prind me tournament selection |
| `_cross` | Bashkon dy prinder me prerje kohore dhe pastaj thirr repair |
| `_mutate` | Largon segmente te dobeta dhe rinderton orarin me repair |
| `_repair` | Mbush boshlleqet kohore me kandidatet me te mire duke ruajtur validitetin |
| `_can_add` | Kontrollon nese nje program mund te shtohet pa shkelur kufizimet |
| `_best_fill_candidate` | Zgjedh kandidatin me te mire per intervalin aktual |
| `_make_child` | Krijon femiun nga dy prinder dhe e krahason me prindin me te mire |
| `generate_solution` | Ekzekuton ciklin kryesor te algoritmit gjenetik deri ne limitin e kohes |

Kjo ndarje e ben algoritmin me te lehte per t'u kuptuar: selection zgjedh prinderit, crossover krijon femijen, mutation krijon ndryshim, repair e ben zgjidhjen valide, ndersa elitism siguron qe zgjidhja me e mire nuk humbet.

### Perditesimi i Popullates

Ne fund te cdo gjenerate krijohet nje popullate e re.
Procesi eshte:

1. Renditen individet sipas fitness.
2. Dy individet me te mire ruhen direkt si elita.
3. Nga gjysma me e mire e popullates krijohet mating pool.
4. Prinderit zgjidhen me tournament selection.
5. Nga prinderit krijohet nje femije me crossover.
6. Femija mund te pesoje mutacion.
7. Repair e ben femijen valid dhe mundohet te mbushe boshlleqet me programe te mira.
8. Femija krahasohet me prindin me te forte; ne popullate futet kandidati me score me te mire.

Kjo do te thote qe femijet e dobet nuk e prishin popullaten, sepse nuk pranohen automatikisht nese jane me te keq se prinderit. Ne te njejten kohe, mutacioni dhe repair krijojne variante te reja qe mund ta kalojne zgjidhjen ekzistuese.

### Fitness

Fitness eshte score total i orarit pas filtrimit dhe validimit te tij.
Per llogaritje perdoren funksionet ekzistuese te projektit:

```text
AlgorithmUtils.filter_valid_schedule
AlgorithmUtils.score_filtered_schedule
```

Kjo siguron qe vleresimi i algoritmit gjenetik te jete i njejte me logjiken e projektit.

### Selection

Per zgjedhjen e prinderve perdoret **tournament selection**.

Ideja:

- zgjidhen disa individe rastesisht nga popullata;
- individi me fitness me te larte fiton turneun;
- fituesi perdoret si prind per krijimin e femijeve.

Ky mekanizem ruan balancen mes:

- shfrytezimit te zgjidhjeve te mira;
- ruajtjes se diversitetit ne popullate.

### Elitism

Ne cdo gjenerate ruhen individet me te mire.
Kjo do te thote qe zgjidhja me e mire nuk humbet edhe nese mutacioni ose crossover prodhojne femije me score me te ulet.

### Crossover

Crossover eshte operatori qe kombinon dy prinder.

Ne kete projekt perdoret crossover me pike kohore:

- zgjedhet nje kohe prerjeje;
- merret pjesa e pare e orarit nga prindi i pare;
- merret pjesa e dyte nga prindi i dyte;
- pastaj thirret repair per ta bere orarin valid dhe per te mbushur boshlleqet.

Ky operator eshte i pershtatshem per problemin tone, sepse orari eshte i varur nga koha.

### Mutation

Mutacioni perdoret per te krijuar ndryshime te reja ne orar.

Ne kete projekt nuk jane perdorur operatoret e ndaluar:

```text
swap, shift, insert, replace
```

Mutacioni yne eshte i tipit:

```text
remove and repair
```

Ai largon nje ose disa programe me cilesi me te dobet dhe pastaj repair mundohet ta rindertoje ate pjese me programe me te mira.

Kur algoritmi ngec per disa gjenerata, perdoret edhe nje mutacion me i drejtuar:

```text
weak-region mutation
```

Ky mutacion identifikon nje zone te dobet te orarit, largon nje bllok te vogel programesh dhe e rinderton ate zone me repair.

### Repair

Repair eshte pjesa kryesore e algoritmit.

Qellimi i tij eshte:

- te filtroje programet jo valide;
- te ruaje kufizimet e problemit;
- te mbushe boshlleqet kohore me kandidatet me te mire;
- te mos lejoje mbivendosje, shkelje te priority blocks ose tejkalim te zhanrit.

Repair perdor nje liste kandidatesh te renditur dhe merr kandidatet me vlere me te mire.
Per diversitet, ndonjehere zgjedh rastesisht nje nga 3 kandidatet me te mire.

### Cache

Per ta bere ekzekutimin me te shpejte, perdoren cache:

- `f_cache` per fitness;
- `add_cache` per kontrollin nese nje program mund te shtohet;
- `quality_cache` per cilesine e segmenteve.

Kjo eshte e rendesishme sepse algoritmi gjenetik vlereson shume individe dhe pa cache ekzekutimi do te ishte me i ngadalte.

## Parametrat e Eksperimenteve te Reja

Eksperimentet e fundit jane ekzekutuar ne menyre automatike nga `main.py`, pa nderhyrje manuale gjate zgjedhjes se instancave.
Qellimi ishte te krahasohen tri konfigurime te algoritmit gjenetik duke perdorur te njejtat 17 instanca dhe te njejtin limit kohor per secilin run.

Komanda e ekzekutimit ishte:

```powershell
python AA_25-26-main/AA_25-26-main/main.py --ga-auto --ga-runs 10 --ga-time 300
```

Kjo komande i ekzekuton keto tri eksperimente:

| Experiment | Profile | Time strategy | Runs | Pershkrim |
|---|---|---|---:|---|
| `exp_same_equal_v2` | `single` | `equal` | 10 | Te gjitha instancat perdorin te njejtet parametra baze |
| `exp_tuned_equal_v2` | `tuned` | `equal` | 10 | Disa instanca perdorin parametra te pershtatur |
| `exp_tuned_equal_mutboost_v2` | `tuned` | `equal` | 10 | Konfigurim tuned me te njejten ndarje kohe dhe me mekanizmat agresive te mutacionit kur algoritmi ngec |

Parametrat baze te eksperimenteve te reja:

| Parameter | Value | Pershkrim |
|---|---:|---|
| population_size | 50 | Numri baze i individeve ne popullate |
| tournament_size | 3 | Numri i individeve qe garojne per zgjedhjen e nje prindi |
| elite_size | 2 | Numri i individeve me te mire qe ruhen direkt |
| crossover_rate | 0.90 | Probabiliteti i kombinimit te dy prinderve |
| mutation_rate | 0.30 | Probabiliteti baze i mutacionit |
| max_generations | 10000 | Kufiri maksimal i gjeneratave |
| candidate_pool_size | 700 | Numri maksimal i kandidateve per repair |
| repair_random_rate | 0.20 | Probabiliteti qe repair te zgjedhe rastesisht nga kandidatet me te mire |

Ne profilin `tuned`, disa instanca perdorin parametra te pershtatur:

| Instance group | population_size | crossover_rate | mutation_rate | Arsyeja |
|---|---:|---:|---:|---|
| australia_iptv, canada_pw, youtube_gold, youtube_premium | 4 | 0.90 | 0.20 | Popullate me e vogel per me shume gjenerata te shpejta |
| china_pw | 12 | 0.90 | 0.30 | Me shume diversitet ne popullate |
| spain_iptv, uk_iptv | 14 | 0.85 | 0.45 | Eksplorim me agresiv per instanca me potencial me te madh |
| Instancat tjera | 50 | 0.90 | 0.30 | Parametrat baze |

### Ndarja e Kohes

Ne keto eksperimente eshte perdorur strategjia `equal`.
Kjo do te thote qe brenda nje run-i, buxheti total prej rreth 300 sekondash ndahet ne menyre pothuajse te barabarte per 17 instancat.

| Time strategy | Pershkrim |
|---|---|
| `equal` | Cdo instance merr nje pjese te ngjashme te kohes totale te run-it |

Kjo qasje e ben krahasimin me te drejte, sepse te tri eksperimentet kane te njejten strukture kohore.
Ndryshimi kryesor vjen nga parametrat dhe nga sjellja e algoritmit gjate mutation, crossover dhe repair.

## Eksperimentet

Eksperimentet e vjetra jane larguar nga kjo permbledhje dhe README mban vetem tri eksperimentet e reja:

```text
exp_same_equal_v2
exp_tuned_equal_v2
exp_tuned_equal_mutboost_v2
```

Output-et ruhen ne:

```text
data/output/genetic_algorithm/experiments
```

Struktura e output-eve per secilin eksperiment eshte:

```text
experiment_name/
  australia_iptv/
    run1_score.json
    run2_score.json
    ...
  youtube_premium/
    run1_score.json
    ...
  run_summaries/
    run1_summary.json
    ...
  summary.json
```

Secili eksperiment ka:

| Item | Value |
|---|---:|
| Runs | 10 |
| Instanca per run | 17 |
| Output-e te pritura | 170 |
| Summary per run | 10 |
| Summary final | 1 |

Total score i zgjidhjeve fillestare nga Branch and Bound mbetet:

```text
136573
```

## Rezultatet Kryesore

Permbledhja e rezultateve per 10 runs:

| Experiment | Best total score | Average total score | Worst total score | Best run | Gain vs BnB |
|---|---:|---:|---:|---:|---:|
| `exp_same_equal_v2` | 148228 | 146782.9 | 145438 | 9 | +11655 |
| `exp_tuned_equal_v2` | 172563 | 166271.8 | 156977 | 7 | +35990 |
| `exp_tuned_equal_mutboost_v2` | 172791 | 165672.7 | 156939 | 2 | +36218 |

Nga kjo tabele shihet qe konfigurimet `tuned` japin permiresim shume me te madh se konfigurimi me parametra te njejte per te gjitha instancat.
Rezultati me i larte i arritur eshte:

```text
Best total score = 172791
```

Ky rezultat vjen nga:

```text
exp_tuned_equal_mutboost_v2, run 2
```

Permiresimi krahasuar me Branch and Bound eshte:

```text
172791 - 136573 = +36218
```

## Rezultatet per Secilin Run

| Run | exp_same_equal_v2 score | Time (s) | exp_tuned_equal_v2 score | Time (s) | exp_tuned_equal_mutboost_v2 score | Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 147041 | 300.39 | 156977 | 286.78 | 172043 | 252.21 |
| 2 | 147340 | 301.06 | 171336 | 246.27 | 172791 | 242.86 |
| 3 | 146880 | 300.27 | 171396 | 283.81 | 166129 | 300.32 |
| 4 | 145438 | 300.63 | 168683 | 280.26 | 156939 | 257.74 |
| 5 | 147352 | 300.28 | 167084 | 279.42 | 165884 | 289.11 |
| 6 | 145835 | 300.71 | 168686 | 300.32 | 170307 | 300.43 |
| 7 | 146160 | 300.73 | 172563 | 298.47 | 168458 | 300.40 |
| 8 | 146689 | 300.55 | 158188 | 292.37 | 158093 | 300.05 |
| 9 | 148228 | 300.36 | 164779 | 284.37 | 164571 | 272.27 |
| 10 | 146866 | 300.59 | 163026 | 300.38 | 161512 | 300.41 |

## Rezultatet me te Mira per Secilen Instance

| Instance | BnB initial | exp_same_equal_v2 best | exp_tuned_equal_v2 best | exp_tuned_equal_mutboost_v2 best | Overall best | Best experiment |
|---|---:|---:|---:|---:|---:|---|
| australia_iptv | 3541 | 4370 | 4343 | 4354 | 4370 | `exp_same_equal_v2` |
| canada_pw | 4157 | 4798 | 4725 | 4725 | 4798 | `exp_same_equal_v2` |
| china_pw | 2535 | 2822 | 2805 | 2801 | 2822 | `exp_same_equal_v2` |
| croatia_tv | 2120 | 2220 | 2220 | 2220 | 2220 | te trija |
| france_iptv | 4046 | 8531 | 9299 | 8499 | 9299 | `exp_tuned_equal_v2` |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | te trija |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | te trija |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | te trija |
| singapore_pw | 4292 | 4617 | 4631 | 4654 | 4654 | `exp_tuned_equal_mutboost_v2` |
| spain_iptv | 4232 | 5344 | 5450 | 5380 | 5450 | `exp_tuned_equal_v2` |
| toy | 510 | 510 | 510 | 510 | 510 | te trija |
| uk_iptv | 4855 | 6075 | 6022 | 6239 | 6239 | `exp_tuned_equal_mutboost_v2` |
| uk_tv | 2171 | 2246 | 2246 | 2246 | 2246 | te trija |
| usa_tv | 3561 | 3575 | 3575 | 3575 | 3575 | te trija |
| us_iptv | 4168 | 4568 | 4569 | 4569 | 4569 | `exp_tuned_equal_v2`, `exp_tuned_equal_mutboost_v2` |
| youtube_gold | 66919 | 67117 | 67421 | 67413 | 67421 | `exp_tuned_equal_v2` |
| youtube_premium | 22641 | 25805 | 49868 | 50033 | 50033 | `exp_tuned_equal_mutboost_v2` |

## Mesatarja per Secilen Instance

| Instance | exp_same_equal_v2 avg | exp_tuned_equal_v2 avg | exp_tuned_equal_mutboost_v2 avg |
|---|---:|---:|---:|
| australia_iptv | 4244.2 | 4226.0 | 4232.0 |
| canada_pw | 4704.0 | 4645.5 | 4646.4 |
| china_pw | 2796.1 | 2793.5 | 2793.0 |
| croatia_tv | 2220.0 | 2220.0 | 2220.0 |
| france_iptv | 8084.4 | 8328.4 | 8038.3 |
| germany_tv | 1626.0 | 1626.0 | 1626.0 |
| kosovo_tv | 2567.0 | 2567.0 | 2567.0 |
| netherlands_tv | 2632.0 | 2632.0 | 2632.0 |
| singapore_pw | 4522.4 | 4528.0 | 4537.2 |
| spain_iptv | 5175.1 | 5288.8 | 5268.6 |
| toy | 510.0 | 510.0 | 510.0 |
| uk_iptv | 5711.9 | 5699.8 | 5711.7 |
| uk_tv | 2241.8 | 2241.2 | 2241.2 |
| usa_tv | 3575.0 | 3575.0 | 3575.0 |
| us_iptv | 4475.6 | 4476.9 | 4476.9 |
| youtube_gold | 67055.0 | 67306.2 | 67296.2 |
| youtube_premium | 24642.4 | 43607.5 | 43301.2 |

## Interpretimi i Rezultateve

Rezultatet tregojne tri gjera kryesore:

1. Parametrat e njejte per te gjitha instancat japin permiresim te qendrueshem, por te kufizuar.
2. Parametrat e pershtatur sipas instances rrisin ndjeshem score-in total.
3. Eksperimenti `exp_tuned_equal_mutboost_v2` arriti score-in me te mire absolut, sidomos fale permiresimeve te medha ne instanca si `youtube_premium`, `uk_iptv` dhe `singapore_pw`.

Disa instanca si `toy`, `germany_tv`, `kosovo_tv` dhe `netherlands_tv` nuk ndryshojne shume, sepse zgjidhja fillestare nga Branch and Bound eshte tashme shume afer maksimumit praktik per ato raste.
Instancat me te medha, sidomos `youtube_premium`, kane hapesire me te madhe per permiresim dhe aty algoritmi gjenetik jep ndikimin me te madh.

## Si te Ekzekutohet Projekti

Per ekzekutim interaktiv:

```powershell
python AA_25-26-main/AA_25-26-main/main.py
```

Per ekzekutimin automatik te tri eksperimenteve te reja:

```powershell
python AA_25-26-main/AA_25-26-main/main.py --ga-auto --ga-runs 10 --ga-time 300
```

Per ekzekutim direkt te nje eksperimenti te vetem me `ga_experiment.py`:

```powershell
python AA_25-26-main/AA_25-26-main/ga_experiment.py --runs 10 --total-time 300 --profile tuned --time-profile equal --experiment-name exp_tuned_equal_v2
```

## Shembull Ekzekutimi me Instance `australia_iptv`

Per ta treguar rrjedhen e algoritmit gjenetik, marrim si shembull instancen `australia_iptv`.

### Hapi 1: Leximi i instances

Fillimisht lexohet file-i hyres:

```text
data/input/australia_iptv.json
```

Nga ky file merren kanalet, programet, kohet e hapjes/mbylljes, preferencat kohore, priority blocks dhe kufizimet tjera.

### Hapi 2: Marrja e zgjidhjes fillestare

Pastaj merret output-i i gatshem nga Branch and Bound:

```text
data/output/branchandboundscheduler/australia_iptv_output_branchandboundscheduler_3541.json
```

Ky output ka score fillestar:

```text
Branch and Bound score = 3541
```

Kjo zgjidhje kthehet ne liste programesh te planifikuara dhe perdoret si individi i pare i popullates.

### Hapi 3: Krijimi i popullates fillestare

Ne rezultatin me te mire per `australia_iptv`, eksperimenti ishte `exp_same_equal_v2`, run 6.
Parametrat ishin:

| Parameter | Value |
|---|---:|
| population_size | 50 |
| tournament_size | 3 |
| elite_size | 2 |
| crossover_rate | 0.90 |
| mutation_rate | 0.30 |
| candidate_pool_size | 700 |
| repair_random_rate | 0.20 |

Popullata fillestare nisi nga score-i i Branch and Bound dhe variante te krijuara me mutation + repair.
Disa nga score-et fillestare ishin:

```text
[3541, 3560, 3328, 3489, 3371, 3543, 3543, 3594, 3563, 3491, ...]
```

Kjo tregon pse diversiteti eshte i dobishem: disa individe jane me te dobet se zgjidhja fillestare, por disa variante fillestare mund te jene menjehere me te mira dhe te hapin rruge per permiresime te tjera.

### Hapi 4: Selection, Crossover, Mutation dhe Repair

Ne cdo gjenerate ndodh kjo rrjedhe:

1. Popullata renditet sipas fitness.
2. Dy individet me te mire ruhen si elita.
3. Prinderit zgjidhen me tournament selection.
4. Crossover kombinon dy prinder sipas nje pike kohore.
5. Mutation largon pjese me cilesi te dobet dhe hap vend per alternativa.
6. Repair e ben orarin valid dhe mbush boshlleqet me kandidatet me te mire.
7. Femija pranohet vetem nese eshte me i mire se prindi me i forte.

Kur algoritmi ngec per disa gjenerata, aktivizohet nje sjellje me agresive: mutacioni largon me shume programe ose nje zone te dobet te orarit dhe pastaj repair e rinderton ate pjese.
Kjo ndihmon qe algoritmi te mos mbetet i bllokuar ne te njejten zgjidhje.

### Hapi 5: Rezultati per `australia_iptv`

Rezultati me i mire i arritur per kete instance ishte:

| Field | Value |
|---|---:|
| Experiment | `exp_same_equal_v2` |
| Run | 6 |
| Generations | 35 |
| Time limit (s) | 17.65 |
| Elapsed (s) | 16.53 |
| Final score | 4370 |
| Gain vs BnB | +829 |

Output-i u ruajt ne:

```text
data/output/genetic_algorithm/experiments/exp_same_equal_v2/australia_iptv/run6_4370.json
```

Ky shembull tregon rrjedhen e plote: zgjidhja fillestare merret nga Branch and Bound, krijohet popullata, zgjedhen prinderit, krijohen femije me crossover, ndryshohen me mutation, riparohen me repair dhe ne fund ruhet zgjidhja me score me te larte.

## Perfundim

Rezultatet e reja tregojne qe algoritmi gjenetik permireson ndjeshem zgjidhjet fillestare te Branch and Bound.
Nga tri eksperimentet e reja, rezultati me i mire absolut u arrit nga:

```text
exp_tuned_equal_mutboost_v2, run 2
```

Me total score:

```text
172791
```

Kjo paraqet permiresim prej:

```text
+36218 krahasuar me totalin fillestar te Branch and Bound
```

Prandaj, per rezultat maksimal ne keto ekzekutime, konfigurimi me i mire eshte `exp_tuned_equal_mutboost_v2`.
Per stabilitet mesatar, `exp_tuned_equal_v2` ka mesatare pak me te larte, ndersa `exp_tuned_equal_mutboost_v2` arriti kulmin me te mire.
