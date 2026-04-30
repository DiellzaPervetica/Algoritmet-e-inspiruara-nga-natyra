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

## Parametrat Final

Parametrat baze te algoritmit gjenetik:

| Parameter | Value | Pershkrim |
|---|---:|---|
| population_size | 8 | Numri i individeve qe ruhen ne popullate ne cdo gjenerate |
| tournament_size | 3 | Numri i individeve qe garojne ne tournament selection per zgjedhjen e nje prindi |
| elite_size | 2 | Numri i individeve me te mire qe kalojne direkt ne gjeneraten tjeter |
| crossover_rate | 0.90 | Probabiliteti qe dy prinder te kombinohen me crossover |
| mutation_rate | 0.25 | Probabiliteti qe nje femije te ndryshohet me mutacion |
| max_generations | 10000 | Kufiri maksimal i gjeneratave; ne praktike algoritmi ndalet nga limiti i kohes |
| candidate_pool_size | 700 | Numri maksimal i kandidateve te ruajtur per secilen kohe fillimi gjate repair |
| repair_random_rate | 0.20 | Probabiliteti qe repair te zgjedhe rastesisht nga kandidatet me te mire per diversitet |

Disa parametra jane mbajtur te njejte per te gjitha instancat:

```text
tournament_size = 3
elite_size = 2
candidate_pool_size = 700
repair_random_rate = 0.20
```

Disa parametra jane pershtatur per disa instanca, sepse gjate eksperimenteve u pa qe japin score me te mire:

| Instance group | population_size | crossover_rate | mutation_rate |
|---|---:|---:|---:|
| australia_iptv, canada_pw, youtube_gold, youtube_premium | 4 | 0.90 | 0.20 |
| china_pw | 12 | 0.90 | 0.30 |
| spain_iptv, uk_iptv | 14 | 0.85 | 0.45 |

Instancat tjera perdorin parametrat baze.

### Arsyetimi i Parametrave

Parametrat nuk jane zgjedhur vetem si vlera fikse, por jane testuar me disa konfigurime. Qellimi ishte te gjendej nje balancim mes score-it dhe kohes se kufizuar prej rreth 300 sekondash per 17 instanca.

Arsyetimi per parametrat e perbashket:

- `tournament_size = 3` jep presion te mjaftueshem selektiv pa e bere algoritmin shume agresiv. Nese turneu eshte shume i madh, popullata humb diversitet shpejt.
- `elite_size = 2` ruan dy zgjidhjet me te mira ne cdo gjenerate. Kjo mbron score-in me te mire nga humbja pas mutacionit ose crossover.
- `candidate_pool_size = 700` mban mjaftueshem kandidate per repair pa e ngadalesuar shume kerkimin. Vlera me e vogel mund te humbe programe te mira, ndersa vlera shume me e madhe rrit kohen e kerkimit.
- `repair_random_rate = 0.20` shton pak diversitet ne repair. Shumicen e kohes merret kandidati me i mire, por ndonjehere zgjidhet nje kandidat tjeter nga top kandidatet per te shmangur ngecjen ne te njejten zgjidhje.

Arsyetimi per parametrat e pershtatur:

- `australia_iptv`, `canada_pw`, `youtube_gold`, `youtube_premium` performuan me mire me `population_size = 4` dhe `mutation_rate = 0.20`, sepse keto instanca kane zgjidhje fillestare te forta ose kerkimi ne to perfiton me shume nga gjenerata te shpejta sesa nga popullate e madhe. Popullata me e vogel lejon me shume gjenerata brenda kohes se njejte.
- `china_pw` u pershtat me `population_size = 12` dhe `mutation_rate = 0.30`, sepse kishte nevoje per me shume diversitet ne popullate. Kjo ndihmon kur zgjidhjet alternative nuk dalin lehte nga mutacione te vogla.
- `spain_iptv` dhe `uk_iptv` u pershtaten me `population_size = 14`, `crossover_rate = 0.85` dhe `mutation_rate = 0.45`, sepse keto instanca perfitonin nga eksplorim me agresiv. Mutacioni me i larte dhe popullata me e madhe krijojne me shume variante, ndersa crossover pak me i ulet e zvogelon rrezikun qe femijet te jene shume te ngjashem me prinderit.

Kjo qasje e ndan problemin ne dy raste:

- instancat ku zgjidhja fillestare eshte tashme shume e mire perdorin parametra me konservativ;
- instancat ku ka me shume hapesire per permiresim perdorin me shume diversitet dhe mutacion.

### Ndarja e Kohes

Per eksperimentet jane testuar dy menyra te ndarjes se kohes:

| Time strategy | Pershkrim |
|---|---|
| equal | Koha ndahet pothuajse barabarte per te gjitha instancat |
| adaptive | Instancat qe kane me shume potencial per permiresim marrin me shume kohe |

Koha adaptive u perdor ne konfigurimin final, sepse jo te gjitha instancat kane te njejten veshtiresi. Disa instanca te vogla, si `toy`, `germany_tv` ose `kosovo_tv`, arrijne shpejt score-in maksimal ose nuk kane shume hapesire per ndryshim. Instanca me te medha ose me me shume potencial, si `france_iptv`, `spain_iptv`, `uk_iptv` dhe `youtube_premium`, perfitojne me shume nga sekonda shtese.

Prandaj koha adaptive e perdor buxhetin total me mire: nuk shpenzon shume kohe ne instanca qe stabilizohen shpejt dhe i jep me shume kohe instancave ku algoritmi mund te rrise score-in.

## Eksperimentet

Per te analizuar ndikimin e parametrave ne cilesine e zgjidhjeve, jane realizuar tri eksperimente kryesore.
Secili eksperiment perdor te njejtat zgjidhje fillestare nga Branch and Bound, ndersa ndryshon menyra e konfigurimit te parametrave ose ndarja e kohes.

Limiti i kohes per secilin eksperiment:

```text
1 run = 17 instanca brenda rreth 300 sekondave
```

| Experiment | Parameter strategy | Time strategy | Runs | Total score | Gain vs BnB |
|---|---|---|---:|---:|---:|
| exp_same_equal | Parametra te njejte per te gjitha instancat | Kohe e barabarte | 1 | 148174 | +11601 |
| exp_tuned_equal | Parametra te pershtatur per disa instanca | Kohe e barabarte | 1 | 153933 | +17360 |
| exp_tuned_adaptive | Parametra te pershtatur per disa instanca | Kohe adaptive | 1 | 166334 | +29761 |

Rezultatet per secilen instance ne tri eksperimentet:

| Instance | BnB initial | exp_same_equal | exp_tuned_equal | exp_tuned_adaptive | Best experiment |
|---|---:|---:|---:|---:|---:|
| australia_iptv | 3541 | 4233 | 4241 | 4241 | 4241 |
| canada_pw | 4157 | 4709 | 4650 | 4650 | 4709 |
| china_pw | 2535 | 2799 | 2805 | 2805 | 2805 |
| croatia_tv | 2120 | 2220 | 2220 | 2220 | 2220 |
| france_iptv | 4046 | 8844 | 8690 | 9472 | 9472 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 |
| singapore_pw | 4292 | 4418 | 4425 | 4406 | 4425 |
| spain_iptv | 4232 | 5113 | 5301 | 5389 | 5389 |
| toy | 510 | 510 | 510 | 510 | 510 |
| uk_iptv | 4855 | 5428 | 5981 | 5981 | 5981 |
| uk_tv | 2171 | 2240 | 2240 | 2240 | 2240 |
| us_iptv | 4168 | 4378 | 4379 | 4379 | 4379 |
| usa_tv | 3561 | 3575 | 3575 | 3575 | 3575 |
| youtube_gold | 66919 | 67101 | 67151 | 67085 | 67151 |
| youtube_premium | 22641 | 25781 | 30940 | 42556 | 42556 |

Nga keto rezultate shihet qe:

- parametrat e njejte jane baseline i thjeshte;
- parametrat e pershtatur japin rezultat me te mire;
- ndarja adaptive e kohes e rrit score-in me shume.

Prandaj konfigurimi final eshte:

```text
Parametra te pershtatur + kohe adaptive
```

## Ekzekutimi Final

Konfigurimi final u ekzekutua 10 here.
Secili run i ekzekuton 17 instancat brenda rreth 5 minutave.

Komanda e ekzekutimit final:

```powershell
python ga_experiment.py --runs 10 --total-time 300 --profile tuned --time-profile adaptive --experiment-name final_tuned --clean-output
```

Output-et finale ruhen ne:

```text
data/output/genetic_algorithm/experiments/final_tuned
```

Struktura e output-eve:

```text
final_tuned/
  australia_iptv/
    run1_4241.json
    run2_4354.json
    ...
  canada_pw/
    run1_4643.json
    ...
  run_summaries/
    run1_summary.json
    ...
  summary.json
```

U krijuan:

```text
170 output-e = 17 instanca x 10 runs
```

Pas kontrollit praktik me funksionet e validimit:

```text
invalid_outputs = 0
```

## Rezultatet Finale per 10 Runs

| Run | Total score | Time (s) |
|---:|---:|---:|
| 1 | 162137 | 300.09 |
| 2 | 167923 | 300.02 |
| 3 | 163244 | 299.95 |
| 4 | 160610 | 299.90 |
| 5 | 167293 | 300.01 |
| 6 | 167470 | 299.97 |
| 7 | 169699 | 299.90 |
| 8 | 162396 | 299.93 |
| 9 | 166422 | 299.82 |
| 10 | 163745 | 299.92 |

Permbledhje:

| Metric | Value |
|---|---:|
| Best total score | 169699 |
| Average total score | 165093.9 |
| Worst total score | 160610 |
| Branch and Bound initial total | 136573 |
| Best gain vs Branch and Bound | +33126 |

## Rezultatet per Secilen Instance

| Instance | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 | Run 10 | Best | Avg | Worst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| australia_iptv | 4241 | 4354 | 4346 | 4228 | 4296 | 4238 | 4276 | 4189 | 4284 | 4242 | 4354 | 4269.4 | 4189 |
| canada_pw | 4643 | 4704 | 4655 | 4513 | 4663 | 4722 | 4715 | 4704 | 4702 | 4725 | 4725 | 4674.6 | 4513 |
| china_pw | 2805 | 2786 | 2801 | 2801 | 2784 | 2793 | 2788 | 2801 | 2799 | 2799 | 2805 | 2795.7 | 2784 |
| croatia_tv | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220.0 | 2220 |
| france_iptv | 9283 | 8628 | 8885 | 8630 | 8888 | 8646 | 8851 | 9446 | 9249 | 8850 | 9446 | 8935.6 | 8628 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626.0 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 2632 |
| singapore_pw | 4406 | 4413 | 4406 | 4420 | 4624 | 4417 | 4555 | 4406 | 4406 | 4555 | 4624 | 4460.8 | 4406 |
| spain_iptv | 5387 | 5417 | 5494 | 5300 | 5506 | 5378 | 5428 | 5529 | 5178 | 5445 | 5529 | 5406.2 | 5178 |
| toy | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510.0 | 510 |
| uk_iptv | 5981 | 6334 | 5372 | 5530 | 6279 | 6122 | 6166 | 6364 | 5767 | 6432 | 6432 | 6034.7 | 5372 |
| uk_tv | 2240 | 2209 | 2240 | 2240 | 2240 | 2209 | 2240 | 2240 | 2246 | 2240 | 2246 | 2234.4 | 2209 |
| usa_tv | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575.0 | 3575 |
| us_iptv | 4379 | 4379 | 4478 | 4359 | 4376 | 4405 | 4528 | 4360 | 4466 | 4318 | 4528 | 4404.8 | 4318 |
| youtube_gold | 67063 | 67016 | 67016 | 66964 | 67021 | 66919 | 66935 | 67054 | 66993 | 66969 | 67063 | 66995.0 | 66919 |
| youtube_premium | 38579 | 44553 | 40421 | 38495 | 43486 | 44491 | 46087 | 38173 | 43202 | 40040 | 46087 | 41752.7 | 38173 |

## Si te Ekzekutohet Projekti

Per ekzekutim interaktiv:

```powershell
python main.py
```

Per ekzekutimin e eksperimenteve te algoritmit gjenetik:

```powershell
python ga_experiment.py --runs 10 --total-time 300 --profile tuned --time-profile adaptive --experiment-name final_tuned --clean-output
```

## Shembull Ekzekutimi me Instance `australia_iptv`

Per ta treguar me qarte se si algoritmi gjenetik e permireson nje zgjidhje, shembulli kryesor merret nga instanca `australia_iptv`.
Kjo instance eshte me e pershtatshme se `toy` per shpjegim, sepse te `toy` zgjidhja fillestare eshte tashme shume e mire dhe score-i nuk ndryshon.

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

Per `australia_iptv`, konfigurimi final perdor:

| Parameter | Value |
|---|---:|
| population_size | 4 |
| tournament_size | 3 |
| elite_size | 2 |
| crossover_rate | 0.90 |
| mutation_rate | 0.20 |

Ne run-in e pare, popullata fillestare kishte keto score:

```text
[3541, 3525, 3510, 3499]
```

Kuptimi i ketyre vlerave:

| Individ | Si krijohet | Score |
|---:|---|---:|
| 1 | Zgjidhja origjinale nga Branch and Bound | 3541 |
| 2 | Variant i krijuar me mutacion dhe repair | 3525 |
| 3 | Variant i krijuar me mutacion dhe repair | 3510 |
| 4 | Variant i krijuar me mutacion dhe repair | 3499 |

Edhe pse disa individe fillojne me score me te ulet, ata jane te dobishem sepse krijojne diversitet. Pa diversitet, crossover dhe mutation do te punonin mbi zgjidhje shume te ngjashme dhe algoritmi do te ngecej me shpejt.

### Hapi 4: Selection

Ne cdo gjenerate, individet renditen sipas fitness.
Dy individet me te mire ruhen si elita dhe nuk humben.

Me `tournament_size = 3`, per zgjedhjen e nje prindi merren disa individe nga popullata dhe fiton ai me score me te larte.
Kjo ben qe zgjidhjet e mira te kene me shume gjasa te perdoren si prinder, por prape mbetet pak rastesi per te ruajtur diversitet.

### Hapi 5: Crossover

Pas zgjedhjes se dy prinderve, aplikohet crossover me pike kohore.
Algoritmi zgjedh nje kohe prerjeje brenda intervalit te dites.

Shembull konceptual:

```text
Parent 1: programet para kohes se prerjes
Parent 2: programet pas kohes se prerjes
Child:    kombinim i dy pjeseve
```

Ky lloj crossover eshte i natyrshem per scheduling, sepse orari eshte i renditur ne kohe. Femija merr nje pjese te dites nga nje prind dhe pjesen tjeter nga prindi tjeter.

### Hapi 6: Mutation

Pas crossover, femija mund te pesoje mutacion.
Mutacioni nuk perdor `swap`, `shift`, `insert` apo `replace`.

Mutacioni yne funksionon keshtu:

1. Identifikon nje program ose nje zone me cilesi me te dobet.
2. E largon ate pjese nga orari.
3. Thirret repair per ta rindertuar ate interval me programe me te mira.

Kjo eshte arsyeja pse mutacioni nuk eshte thjesht ndryshim i rastesishem. Ai mundohet te hape hapesire ne pjeset ku orari ka potencial per permiresim.

### Hapi 7: Repair

Repair eshte hapi qe e ben femijen valid.
Ai kontrollon qe programet:

- te jene brenda opening time dhe closing time;
- te mos mbivendosen;
- te respektojne minimum duration;
- te respektojne priority blocks;
- te mos e kalojne max consecutive genre;
- te mos perdoren dy here te njejtat programe.

Pas kesaj, repair mundohet te mbushe boshlleqet me kandidatet me te mire.
Kandidatet zgjidhen sipas score-it baze, bonusit te mundshem dhe penaliteteve si pritja ose nderrimi i kanalit.

### Hapi 8: Pranimi i femijes dhe perditesimi i popullates

Femija i krijuar nuk futet automatikisht ne popullate.
Ai krahasohet me prindin me te forte.

```text
Nese child eshte me i mire -> futet child
Nese child eshte me i dobet -> ruhet prindi me i forte
```

Kjo e mbron algoritmin nga humbja e zgjidhjeve te mira, ndersa prape lejon futjen e femijeve kur ata sjellin permiresim.

### Hapi 9: Rezultati per `australia_iptv`

Ne run-in e pare, algoritmi beri:

```text
generations = 592
time_limit = 20.0 seconds
final_score = 4241
```

Permiresimi ndaj zgjidhjes fillestare:

```text
4241 - 3541 = +700
```

Output-i final u ruajt ne:

```text
data/output/genetic_algorithm/experiments/final_tuned/australia_iptv/run1_4241.json
```

Ky shembull tregon rrjedhen e plote te algoritmit: zgjidhja fillestare merret nga Branch and Bound, krijohet popullata, zgjidhen prinderit, krijohen femije me crossover, ndryshohen me mutation, riparohen me repair dhe ne fund ruhet zgjidhja me score me te larte.

Per krahasim, `toy` mbetet shembull i thjeshte ku score-i eshte `510` ne te gjitha run-et, sepse hapesira e problemit eshte shume e vogel dhe Branch and Bound e gjen menjehere zgjidhjen me te mire.

## Perfundim

Rezultatet tregojne qe algoritmi gjenetik permireson ndjeshem zgjidhjet fillestare te Branch and Bound.
Strategjia me e mire ishte perdorimi i zgjidhjeve te gatshme nga Branch and Bound si warm start, pastaj aplikimi i algoritmit gjenetik me parametra te pershtatur dhe kohe adaptive.

Konfigurimi final arriti:

```text
Best total score = 169699
```

Kjo paraqet nje permiresim prej:

```text
+33126 krahasuar me totalin fillestar te Branch and Bound
```
