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

## Përmbajtja

- [Përshkrimi i Projektit](#përshkrimi-i-projektit-optimizimi-i-orarit-televiziv)
- [Zgjidhjet Fillestare](#zgjidhjet-fillestare)
- [Beam Search Scheduler](#beam-search-scheduler)
- [Branch and Bound Scheduler](#branch-and-bound-scheduler)
- [Algoritmi Gjenetik](#algoritmi-gjenetik)
- [Struktura e Algoritmit Gjenetik](#struktura-e-algoritmit)
- [Parametrat e Algoritmit Gjenetik](#parametrat-e-algoritmit-gjenetik)
- [Eksperimentet e Realizuara](#eksperimentet-e-realizuara)
- [Parametrat e Eksperimenteve](#parametrat-sipas-eksperimentit)
- [Koha e Ekzekutimit](#koha-e-ekzekutimit)
- [Rezultatet e Eksperimenteve](#rezultatet-e-eksperimenteve)
- [Rezultatet e Plota për çdo Eksperiment](#rezultatet-e-plota-për-çdo-eksperiment)
- [Konfigurimi Final i Zgjedhur](#konfigurimi-final-i-zgjedhur)
- [Përfundim i Algoritmit Gjenetik](#përfundim-i-algoritmit-gjenetik)
- [Përmirësimi Final me Iterated Local Search](#përmirësimi-final-me-iterated-local-search)
- [Rezultatet e GA + ILS](#rezultatet-e-ga--ils)
- [Rezultatet e Plota te GA + ILS](#rezultatet-e-plota-te-ga--ils)
- [Si të Ekzekutohet Projekti](#si-të-ekzekutohet-projekti)


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

Para algoritmit gjenetik janë implementuar dy metoda për gjenerimin e zgjidhjeve fillestare:

```text
1. Beam Search Scheduler
2. Branch and Bound Scheduler
```

Këto metoda shërbejnë si bazë krahasimi dhe si mënyrë për të krijuar zgjidhje fillestare të mira.

## Beam Search Scheduler

Beam Search Scheduler është një metodë heuristike që mban disa zgjidhje të pjesshme më të mira në çdo hap, në vend që të ndjekë vetëm një zgjidhje të vetme.

Ideja kryesore:

- ndërtohet orari hap pas hapi;
- në çdo hap ruhen disa kandidatë më të mirë;
- përdoret një vlerësim heuristik për të zgjedhur drejtimet më premtuese;
- shmanget një pjesë e gabimeve që mund t'i bëjë një algoritm greedy.

Parametrat kryesore të Beam Search:

| Parameter | Value | Përshkrim |
|---|---:|---|
| beam_width | 100 | Numri i zgjidhjeve të pjesshme që ruhen në çdo hap |
| lookahead_limit | 4 | Sa hapa përpara analizohet ndikimi i një vendimi |

Beam Search është përdorur si metodë fillestare dhe si pikë krahasimi, por për algoritmin gjenetik kemi vendosur të përdorim Branch and Bound si zgjidhje iniciale, sepse output-et e tij kanë qenë më të përshtatshme për warm start.

## Branch and Bound Scheduler

Branch and Bound Scheduler eksploron hapësirën e zgjidhjeve duke krijuar degëzime të mundshme dhe duke larguar drejtimet që nuk duken premtuese.
Kjo metodë është përdorur për të krijuar zgjidhje fillestare të forta, të cilat më pas shërbejnë si pikë nisjeje për algoritmin gjenetik.

Ideja kryesore:

- ndërtohen zgjidhje të pjesshme duke ecur nëpër kohë;
- në çdo pikë vendimi gjenerohen programet kandidate që mund të shtohen në orar;
- për çdo kandidat llogaritet score-i duke përfshirë score-in bazë, bonuset, penalitetet dhe kufizimet;
- për çdo degëzim vlerësohet potenciali maksimal që mund të arrihet me pjesën e mbetur të ditës;
- degëzimet që nuk mund të japin rezultat më të mirë se zgjidhja aktuale më e mirë largohen;
- ruhet zgjidhja me score më të lartë që gjendet brenda kohës së lejuar.

Komponentet kryesore të implementimit:

| Komponenti | Qëllimi |
|---|---|
| Preprocessing | Rendit programet sipas kanaleve dhe koheve, pergatit priority blocks dhe vlerat optimiste për pruning |
| Randomized warm starts | Krijon disa zgjidhje fillestare të shpejta për të pasur një incumbent të mirë para kërkimit kryesor |
| Candidate generation | Gjen programet që mund të vendosen në kohën aktuale pa shkelur kufizimet |
| Upper bound | Vlereson kufirin maksimal teorik që mund të arrihet nga një gjendje e pjesshme |
| Pruning | Ndal eksplorimin e degëzimeve që nuk mund ta kalojnë score-in më të mirë aktual |
| DFS search | Eksploron degëzimet me stack dhe përditëson zgjidhjen më të mirë |

Arsyeja pse kjo metodë është e përshtatshme për warm start është se zakonisht prodhon zgjidhje valide dhe me score të mirë. Algoritmi gjenetik nuk fillon nga një orar plotësisht random, por nga një zgjidhje që tashmë respekton kufizimet kryesore dhe ka score të lartë.

Output-et e Branch and Bound ruhen në:

```text
data/output/branchandboundscheduler
```

Në algoritmin gjenetik, këto output-e janë përdorur si **zgjidhje iniciale të gatshme**.

E rëndësishme:

```text
Algoritmi gjenetik nuk e gjeneron prapë zgjidhjen fillestare me Branch and Bound.
Ai e lexon zgjidhjen ekzistuese nga output-et e ruajtura të Branch and Bound.
```

Kjo qasje u zgjodh për dy arsye:

- kursen kohë gjatë ekzekutimit të algoritmit gjenetik;
- e bën eksperimentin më të kontrollueshëm, sepse çdo run niset nga e njëjta zgjidhje iniciale për atë instance.

Total score i zgjidhjeve fillestare Branch and Bound për 17 instancat ishte:

```text
136573
```

## Algoritmi Gjenetik

Algoritmi gjenetik përdoret për të përmirësuar zgjidhjet fillestare të marra nga Branch and Bound. Në këtë projekt, një individ në popullatë përfaqëson një orar të plotë ose të pjesshëm:

```text
individual = liste e programeve të planifikuara
```

Popullata fillestare nuk krijohet nga zero. Si individi i parë merret zgjidhja e gatshme nga Branch and Bound, ndërsa individët tjerë krijohen duke bërë mutacion mbi këtë zgjidhje dhe pastaj duke aplikuar `repair`. Kjo qasje e ruan një pikënisje të fortë dhe, në të njëjtën kohë, krijon diversitet për kërkim.

### Struktura e Algoritmit

Rrjedha kryesore e algoritmit është:

1. Lexohet instanca nga `data/input`.
2. Lexohet output-i përkatës nga `data/output/branchandboundscheduler`.
3. Krijohet popullata fillestare nga zgjidhja Branch and Bound.
4. Llogaritet fitness për secilin individ.
5. Zgjidhen prindërit me tournament selection.
6. Krijohen fëmijë me crossover.
7. Aplikohet mutacioni.
8. Aplikohet repair për ta mbajtur orarin valid.
9. Popullata përditësohet duke ruajtur elitat dhe kandidatët më të mirë.
10. Ruhet gjithmonë zgjidhja më e mirë e gjetur deri në atë moment.

### Funksionet Kryesore

| Funksioni | Roli |
|---|---|
| `_base_schedule` | Merr zgjidhjen fillestare nga Branch and Bound dhe e filtron që të jetë valide |
| `_initial_population` | Krijon popullatën fillestare duke ruajtur bazën dhe duke krijuar variante me mutacion |
| `_fitness` | Llogarit score-in e një individi pas filtrimit valid |
| `_select_population` | Rendit popullatën, ruan elitat dhe krijon mating pool |
| `_pick_parent` | Zgjedh një prind me tournament selection |
| `_cross` | Bashkon dy prindër me prerje kohore dhe pastaj thërret repair |
| `_mutate` | Largon segmente të dobëta dhe rindërton orarin me repair |
| `_repair` | Mbush boshllëqet kohore me kandidatët më të mirë duke ruajtur validitetin |
| `_can_add` | Kontrollon nëse një program mund të shtohet pa shkelur kufizimet |
| `_best_fill_candidate` | Zgjedh kandidatin më të mirë për intervalin aktual |
| `_make_child` | Krijon fëmijën nga dy prindër dhe e krahason me prindin më të mirë |
| `generate_solution` | Ekzekuton ciklin kryesor të algoritmit gjenetik deri në limitin e kohës |

### Selection, Crossover, Mutation dhe Repair

Për selection përdoret `tournament selection`. Në çdo zgjedhje merren disa individë nga popullata dhe fiton ai me fitness më të lartë. Kjo e favorizon cilësinë, por nuk e largon plotësisht rastësinë, prandaj popullata ruan diversitet.

Crossover-i është me pikë kohore: një pjesë e orarit merret nga prindi i parë dhe pjesa tjetër nga prindi i dytë. Ky operator është i përshtatshëm për scheduling, sepse orari është i renditur në kohë dhe një prerje kohore krijon fëmijë që kanë kuptim për problemin.

Mutacioni nuk përdor operatorët `swap`, `shift`, `insert` ose `replace`. Operatori ynë është `remove and repair`: largohen një ose disa segmente me cilësi më të dobët dhe pastaj `repair` rindërton boshllëkun me kandidatë më të mirë. Kur algoritmi ngec, përdoret edhe një mutacion më agresiv mbi një zonë të dobët të orarit.

`Repair` është pjesa kryesore që e mban zgjidhjen valide. Ai kontrollon kohën e hapjes/mbylljes, mbivendosjet, minimum duration, priority blocks dhe kufizimin për zhanret e njëpasnjëshme. Pastaj mbush boshllëqet me programet më të mira të mundshme sipas score-it, bonuseve dhe penaliteteve.

### Elitism dhe Përditësimi i Popullatës

Në çdo gjeneratë ruhen individët më të mirë (`elite_size`). Këta kalojnë direkt në gjeneratën tjetër, prandaj zgjidhja më e mirë nuk humbet edhe nëse crossover ose mutation prodhojnë fëmijë më të dobët.

Fëmijët krijohen nga mating pool, jo vetëm nga elitat. Pas krijimit, fëmija krahasohet me prindin më të fortë. Nëse fëmija është më i mirë, pranohet fëmija; nëse jo, ruhet prindi më i mirë. Kjo e mbron popullatën nga përkeqësimi, por prapë lejon përmirësime kur fëmija sjell kombinim më të mirë.

### Cache dhe Shpejtësia

Për shkak se algoritmi vlerëson shumë individë, janë përdorur cache për operacionet më të shpeshta:

| Cache | Qëllimi |
|---|---|
| `f_cache` | Ruan fitness-in e orareve që janë vlerësuar më parë |
| `add_cache` | Ruan kontrollin nëse një program mund të shtohet në një gjendje të caktuar |
| `quality_cache` | Ruan cilësinë e segmenteve për mutacionin e drejtuar |

Kjo e bën ekzekutimin më të lehtë për laptop dhe shmang llogaritjet e përsëritura.

## Parametrat e Algoritmit Gjenetik

Parametrat bazë janë përdorur në eksperimentin e parë dhe shërbejnë si konfigurim i balancuar:

| Parameter | Value | Përshkrim |
|---|---:|---|
| `population_size` | 50 | Numri i individeve në popullatë |
| `tournament_size` | 3 | Numri i individeve që garojne për zgjedhjen e një prindi |
| `elite_size` | 2 | Numri i individeve më të mirë që ruhen direkt |
| `crossover_rate` | 0.90 | Probabiliteti që dy prindër të kombinohen me crossover |
| `mutation_rate` | 0.30 | Probabiliteti bazë që një fëmijë të ndryshohet me mutation |
| `max_generations` | 10000 | Kufiri maksimal i gjeneratave, edhe pse zakonisht ndalet nga koha ose stagnimi |
| `candidate_pool_size` | 700 | Numri maksimal i kandidatëve të ruajtur për çdo kohë fillimi gjatë repair |
| `repair_random_rate` | 0.20 | Probabiliteti që repair të zgjedhë rastësisht nga kandidatët më të mirë |
| `REPAIR_TOP_K` | 3 | Numri i kandidatëve më të mirë që mbahen gjatë zgjedhjes në repair |
| `stagnation_limit` | 10 | Ndalesa e hershme nëse nuk ka përmirësim për 10 gjenerata |
| `min_runtime_before_stop` | 20s | Algoritmi nuk ndalet nga stagnimi pa kaluar se paku 20 sekonda |
| `max_instance_time` | 300s | çdo run për çdo instance ka maksimum 5 minuta |

Stagnimi përdoret që algoritmi të mos shpenzoje kohë kur popullata nuk po përmirësohet. Nëse nuk ka përmirësim, pas disa gjeneratave aktivizohet mutacion më agresiv; nëse prapë nuk ka përmirësim dhe kanë kaluar se paku 20 sekonda, ekzekutimi i asaj instance ndalet më herët.

## Eksperimentet e Realizuara

Janë realizuar tri eksperimente. Të tria përdorin të njëjtat 17 instanca dhe të njëjtat zgjidhje fillestare nga Branch and Bound. Ndryshimi është të parametrat e algoritmit gjenetik.

| Experiment | Profile | Ideja kryesore |
|---|---|---|
| `exp_uniform_balanced` | `single` | Parametra të njëjtë për çdo instance, si baseline i pastër dhe i lehtë për krahasim |
| `exp_tuned_by_instance` | `tuned` | Parametra të përshtatur sipas madhësisë së instancës |
| `exp_stronger_exploration` | `strong` | Parametra më eksplorues, me popullatë dhe mutacion më të madh |

### Pse u zgjodhën këto eksperimente

Eksperimenti i parë teston nëse një konfigurim i vetëm mund të funksionojë mirë për të gjitha instancat. Kjo është qasja më e thjeshtë dhe më e lehtë për t'u shpjeguar.

Eksperimenti i dytë e ndan problemin sipas madhësisë së instancës. Instancat e vogla nuk kanë nevojë për popullatë shumë të madhe, ndërsa instancat e mëdha kanë më shumë kombinime të mundshme dhe mund të përfitojnë nga më shumë diversitet.

Eksperimenti i tretë teston nëse eksplorimi më agresiv jep rezultate më të larta. Ai përdor popullatë më të madhe dhe mutation rate më të lartë, por kjo mund të këtë edhe rrezik: nëse zgjidhja fillestare nga Branch and Bound është shumë e mirë, mutacioni tepër agresiv mund ta prishë strukturën e mirë të saj.

### Parametrat sipas Eksperimentit

| Experiment | Parametrat |
|---|---|
| `exp_uniform_balanced` | `population_size=50`, `crossover_rate=0.90`, `mutation_rate=0.30` për të gjitha instancat |
| `exp_tuned_by_instance` | **Small:** `population_size=25`, `mutation_rate=0.25`<br>**Medium:** `population_size=40`, `mutation_rate=0.30`<br>**Large:** `population_size=60`, `crossover_rate=0.88`, `mutation_rate=0.35`<br>**Huge:** `population_size=70`, `crossover_rate=0.85`, `mutation_rate=0.40` |
| `exp_stronger_exploration` | **Small:** `population_size=35`, `mutation_rate=0.30`<br>**Medium:** `population_size=55`, `mutation_rate=0.35`<br>**Large:** `population_size=75`, `crossover_rate=0.85`, `mutation_rate=0.42`<br>**Hard/Huge:** `population_size=85`, `crossover_rate=0.85`, `mutation_rate=0.45` |

Parametrat që mbeten të njëjtë në të gjitha eksperimentet janë:

```text
tournament_size = 3
elite_size = 2
max_generations = 10000
candidate_pool_size = 700
repair_random_rate = 0.20
REPAIR_TOP_K = 3
stagnation_limit = 10
min_runtime_before_stop = 20 seconds
max_instance_time = 300 seconds
```

### Grupimi i Instancave për Parametrat e Përshtatur

| Group | Instances | Arsyeja |
|---|---|---|
| Small | `croatia_tv`, `germany_tv`, `kosovo_tv`, `netherlands_tv`, `toy` | Pak programe dhe hapësirë më e vogël kërkimi |
| Medium | `singapore_pw`, `spain_iptv`, `uk_tv` | Madhesi mesatare, kerkon pak më shumë diversitet |
| Large | `australia_iptv`, `canada_pw`, `france_iptv`, `uk_iptv` | Shumë programe dhe më shumë kombinime të mundshme |
| Huge/Hard | `china_pw`, `usa_tv`, `us_iptv`, `youtube_gold`, `youtube_premium` | Hapësirë shumë e madhe kërkimi ose horizonte më të gjata kohore |

### Koha e Ekzekutimit

çdo run i një instance në një eksperiment ka limit maksimal prej 300 sekondash:

```text
1 run për 1 instance = maksimum 5 minuta
```

Për secilin eksperiment janë bërë 10 runs për secilën nga 17 instancat:

```text
17 instanca x 10 runs = 170 output-e për eksperiment
3 eksperimente x 170 output-e = 510 output-e gjithsej
```

Disa ekzekutime përfundojnë më herët për shkak të early stopping. Kjo ndodh vetëm kur algoritmi ka kaluar kohën minimale prej 20 sekondash dhe nuk ka gjetur përmirësim për disa gjenerata.

## Rezultatet e Eksperimenteve

Total score i zgjidhjeve fillestare nga Branch and Bound është:

```text
136573
```

### Përmbledhje e Rezultateve

| Experiment | Strategjia | Profile | Runs | Best total | Avg total | Worst total | Best run | Gain vs BnB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `exp_uniform_balanced` | Parametra të njëjtë për të gjitha instancat | single | 10 | 166067 | 156936.1 | 148869 | 2 | +29494 |
| `exp_tuned_by_instance` | Parametra të përshtatur sipas madhësisë së instancës | tuned | 10 | 156085 | 154732.8 | 152678 | 3 | +19512 |
| `exp_stronger_exploration` | Parametra të përshtatur + eksplorim më i fortë | strong | 10 | 154423 | 152535.8 | 151503 | 2 | +17850 |

### Total Score për Secilin Run

| Run | `exp_uniform_balanced` | `exp_tuned_by_instance` | `exp_stronger_exploration` |
| ---: | ---: | ---: | ---: |
| 1 | 164721 | 153900 | 153945 |
| 2 | 166067 | 152678 | 154423 |
| 3 | 157212 | 156085 | 151503 |
| 4 | 155877 | 155159 | 151988 |
| 5 | 154499 | 154807 | 152557 |
| 6 | 156822 | 154850 | 151974 |
| 7 | 155556 | 154882 | 152026 |
| 8 | 157563 | 155570 | 151897 |
| 9 | 148869 | 155613 | 152648 |
| 10 | 152175 | 153784 | 152397 |

## Rezultatet e Plota për çdo Eksperiment

### Eksperimenti 1: `exp_uniform_balanced`

Parametra të njëjtë për të gjitha instancat.

| Instance | BnB | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 | Run 10 | Best | Avg | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| australia_iptv | 3541 | 4374 | 4448 | 4395 | 4397 | 4456 | 4396 | 4350 | 4333 | 4344 | 4379 | 4456 | 4387.2 | 4333 |
| canada_pw | 4157 | 4716 | 4775 | 4718 | 4764 | 4741 | 4732 | 4739 | 4722 | 4822 | 4765 | 4822 | 4749.4 | 4716 |
| china_pw | 2535 | 2801 | 2801 | 2827 | 2796 | 2801 | 2805 | 2805 | 2801 | 2796 | 2810 | 2827 | 2804.3 | 2796 |
| croatia_tv | 2120 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220.0 | 2220 |
| france_iptv | 4046 | 9182 | 8854 | 9331 | 9288 | 9730 | 9576 | 9109 | 9527 | 8608 | 9461 | 9730 | 9266.6 | 8608 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626.0 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 2632 |
| singapore_pw | 4292 | 4406 | 4648 | 4555 | 4579 | 4417 | 4740 | 4493 | 4623 | 4659 | 4418 | 4740 | 4553.8 | 4406 |
| spain_iptv | 4232 | 5195 | 5501 | 5192 | 5415 | 5305 | 5480 | 5402 | 5433 | 5450 | 5169 | 5501 | 5354.2 | 5169 |
| toy | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510.0 | 510 |
| uk_iptv | 4855 | 6124 | 6350 | 5942 | 5481 | 5392 | 6211 | 6071 | 6408 | 6170 | 6091 | 6408 | 6024.0 | 5392 |
| uk_tv | 2171 | 2236 | 2240 | 2240 | 2209 | 2240 | 2209 | 2236 | 2231 | 2240 | 2240 | 2240 | 2232.1 | 2209 |
| usa_tv | 3561 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575.0 | 3575 |
| us_iptv | 4168 | 4564 | 4422 | 4529 | 4569 | 4467 | 4569 | 4567 | 4490 | 4532 | 4527 | 4569 | 4523.6 | 4422 |
| youtube_gold | 66919 | 67513 | 67576 | 67445 | 67312 | 67315 | 67284 | 67285 | 67335 | 67543 | 67161 | 67576 | 67376.9 | 67161 |
| youtube_premium | 22641 | 40480 | 41322 | 32908 | 31937 | 30505 | 31690 | 31369 | 32530 | 24575 | 28024 | 41322 | 32534.0 | 24575 |

### Eksperimenti 2: `exp_tuned_by_instance`

Parametra të përshtatur sipas madhësisë së instancës.

| Instance | BnB | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 | Run 10 | Best | Avg | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| australia_iptv | 3541 | 4148 | 4460 | 4371 | 4378 | 4242 | 4307 | 4339 | 4440 | 4404 | 4456 | 4460 | 4354.5 | 4148 |
| canada_pw | 4157 | 4677 | 4728 | 4702 | 4662 | 4778 | 4749 | 4751 | 4728 | 4744 | 4887 | 4887 | 4740.6 | 4662 |
| china_pw | 2535 | 2801 | 2805 | 2801 | 2796 | 2805 | 2801 | 2822 | 2801 | 2801 | 2805 | 2822 | 2803.8 | 2796 |
| croatia_tv | 2120 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220.0 | 2220 |
| france_iptv | 4046 | 10121 | 9465 | 9237 | 9373 | 8305 | 9397 | 9032 | 9434 | 9386 | 8962 | 10121 | 9271.2 | 8305 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626.0 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 2632 |
| singapore_pw | 4292 | 4508 | 4450 | 4738 | 4728 | 4432 | 4650 | 4507 | 4511 | 4752 | 4520 | 4752 | 4579.6 | 4432 |
| spain_iptv | 4232 | 5237 | 5368 | 5415 | 5178 | 5459 | 5225 | 5448 | 5328 | 5381 | 5560 | 5560 | 5359.9 | 5178 |
| toy | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510.0 | 510 |
| uk_iptv | 4855 | 6414 | 6181 | 6121 | 6540 | 6180 | 5840 | 5689 | 5813 | 6265 | 6026 | 6540 | 6106.9 | 5689 |
| uk_tv | 2171 | 2240 | 2240 | 2240 | 2240 | 2240 | 2246 | 2240 | 2240 | 2240 | 2240 | 2246 | 2240.6 | 2240 |
| usa_tv | 3561 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575.0 | 3575 |
| us_iptv | 4168 | 4531 | 4528 | 4569 | 4530 | 4398 | 4417 | 4533 | 4415 | 4492 | 4492 | 4569 | 4490.5 | 4398 |
| youtube_gold | 66919 | 67245 | 67177 | 67269 | 67265 | 67249 | 67266 | 67245 | 67274 | 67278 | 67245 | 67278 | 67251.3 | 67177 |
| youtube_premium | 22641 | 28848 | 28146 | 31492 | 30339 | 31589 | 30822 | 31146 | 31456 | 30740 | 29461 | 31589 | 30403.9 | 28146 |

### Eksperimenti 3: `exp_stronger_exploration`

Parametra të përshtatur + eksplorim më i fortë.

| Instance | BnB | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 | Run 10 | Best | Avg | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| australia_iptv | 3541 | 4376 | 4384 | 4517 | 4484 | 4403 | 4192 | 4386 | 4339 | 4361 | 4422 | 4517 | 4386.4 | 4192 |
| canada_pw | 4157 | 4749 | 4710 | 4741 | 4696 | 4774 | 4660 | 4747 | 4676 | 4697 | 4807 | 4807 | 4725.7 | 4660 |
| china_pw | 2535 | 2801 | 2796 | 2801 | 2801 | 2794 | 2801 | 2801 | 2801 | 2818 | 2796 | 2818 | 2801.0 | 2794 |
| croatia_tv | 2120 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220.0 | 2220 |
| france_iptv | 4046 | 9092 | 9246 | 9269 | 9337 | 10050 | 9338 | 9663 | 9152 | 9600 | 9519 | 10050 | 9426.6 | 9092 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626.0 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 2632 |
| singapore_pw | 4292 | 4624 | 4600 | 4480 | 4618 | 4487 | 4630 | 4617 | 4593 | 4624 | 4480 | 4630 | 4575.3 | 4480 |
| spain_iptv | 4232 | 5379 | 5500 | 5399 | 5330 | 5359 | 5383 | 5354 | 5151 | 5128 | 5331 | 5500 | 5331.4 | 5128 |
| toy | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510.0 | 510 |
| uk_iptv | 4855 | 5886 | 6136 | 5998 | 5780 | 6495 | 6353 | 5859 | 6497 | 6247 | 5998 | 6497 | 6124.9 | 5780 |
| uk_tv | 2171 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2246 | 2240 | 2246 | 2240.6 | 2240 |
| usa_tv | 3561 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575.0 | 3575 |
| us_iptv | 4168 | 4569 | 4531 | 4466 | 4451 | 4565 | 4530 | 4405 | 4532 | 4533 | 4414 | 4569 | 4499.6 | 4405 |
| youtube_gold | 66919 | 67221 | 67232 | 67137 | 67137 | 67172 | 67189 | 67152 | 67172 | 67179 | 67174 | 67232 | 67176.5 | 67137 |
| youtube_premium | 22641 | 29878 | 29918 | 27325 | 27984 | 27088 | 27528 | 27672 | 27614 | 28085 | 28086 | 29918 | 28117.8 | 27088 |

## Interpretimi i Rezultateve

Nga rezultatet shihet që të tria konfigurimet përmirësojnë zgjidhjen fillestare nga Branch and Bound. Megjithate, në këto ekzekutime konkrete, rezultati më i mirë total dhe mesatarja më e mirë u arriten nga `exp_uniform_balanced`.

Kjo tregon që për këtë problem, një konfigurim i balancuar me `population_size=50`, `crossover_rate=0.90` dhe `mutation_rate=0.30` ruan raport më të mirë mes eksplorimit dhe stabilitetit. Konfigurimet e përshtatura dhe më agresive krijojnë më shumë diversitet, por në disa instanca e prishin më shpesh strukturën e mirë të zgjidhjes fillestare nga Branch and Bound.

Eksperimenti `exp_tuned_by_instance` dhe `exp_stronger_exploration` prapë janë të vlefshme për analize, sepse tregojne efektin e parametrave të ndryshëm. Në disa instanca ato arrijne score të mirë, por në total nuk e kalojnë konfigurimin e parë.

## Konfigurimi Final i Zgjedhur

Për versionin final zgjedhim:

```text
exp_uniform_balanced
```

Arsyeja është se ky eksperiment dha:

| Metric | Value |
|---|---:|
| Best total score | 166067 |
| Average total score | 156936.1 |
| Worst total score | 148869 |
| Best gain vs Branch and Bound | +29494 |

Pra, edhe pse janë testuar parametra të përshtatur dhe eksplorim më i fortë, konfigurimi i balancuar me parametra të njëjtë për të gjitha instancat doli më i qëndrueshëm dhe më i suksesshëm në total.

## Përfundim i Algoritmit Gjenetik

Algoritmi gjenetik u ndërtua mbi zgjidhjet iniciale të Branch and Bound dhe përdori selection, crossover, mutation, repair, elitism, cache dhe early stopping për të kërkuar orare me score më të lartë.

Rezultati më i mirë final ishte:

```text
166067
```

Ky rezultat është përmirësim ndaj Branch and Bound për:

```text
166067 - 136573 = +29494
```

Kjo e bën `exp_uniform_balanced` konfigurimin final më të mirë në rezultatet aktuale të projektit.

## Përmirësimi Final me Iterated Local Search

Pas perfundimit te eksperimenteve me algoritmin gjenetik, eshte shtuar edhe nje faze e dyte optimizimi me **Iterated Local Search**. Qellimi i kesaj faze nuk eshte te krijoje orare nga zero, por te marre zgjidhjet me te mira te eksperimentit `exp_uniform_balanced` dhe te kerkoje permiresime shtese mbi to.

Pra, per secilen instance merret si seed rezultati me i mire i gjetur nga:

```text
data/output/genetic_algorithm/experiments/exp_uniform_balanced
```

Pastaj mbi kete seed aplikohet Iterated Local Search. Output-et finale ruhen ne:

```text
data/output/ga_and_ils
```

E rendesishme:

```text
GA + ILS nuk e ekzekuton prape algoritmin gjenetik.
Faza ILS lexon rezultatin me te mire ekzistues nga exp_uniform_balanced dhe e perdor si pikenisje.
```

### Ideja Kryesore e ILS

Iterated Local Search punon duke mbajtur gjithmone zgjidhjen me te mire aktuale dhe duke provuar ndryshime te kontrolluara mbi te. Nese nje ndryshim jep score me te mire, zgjidhja pranohet. Nese jo, zgjidhja me e mire ruhet dhe kerkimi vazhdon deri ne stagnim ose deri ne limitin kohor.

Rrjedha kryesore eshte:

1. Lexohet instanca nga `data/input`.
2. Gjehet output-i me score me te larte nga `exp_uniform_balanced`.
3. Ky output perdoret si seed fillestar.
4. Aplikohet local search mbi seed-in.
5. Behet perturbim i kontrolluar i orarit.
6. Aplikohet repair per ta kthyer orarin ne gjendje valide.
7. Kandidati pranohet vetem nese ka score me te larte se zgjidhja me e mire aktuale.
8. Nese kandidati nuk sjell permiresim, rritet stagnimi.
9. Stagnimi nuk resetohet ne zero pas permiresimit; ai mbetet numer i tentimeve te pasuksesshme.
10. Ruhet gjithmone rezultati me i mire i gjetur.

Kjo qasje u zgjodh sepse seed-i nga algoritmi gjenetik eshte tashme i forte. Prandaj, ne kete faze nuk kerkohet eksplorim shume agresiv, por permiresim lokal i kujdesshem.

### Funksionet Kryesore

| Funksioni | Roli |
|---|---|
| `find_best_ga_output` | Gjen output-in me score me te larte nga eksperimenti `exp_uniform_balanced` |
| `ILS` | Klasa qe ekzekuton Iterated Local Search per nje instance |
| `load_initial_schedule` | Lexon output-in ekzistues te GA dhe e kthen ne orar te perdorshem |
| `save_solution` | Ruan zgjidhjen e re ne format JSON |
| `_program_value` | Vlereson kandidatin duke kombinuar score-in e programit me bonusin maksimal te zhanrit |
| `_score` | Llogarit score-in e orarit pas filtrimit valid |
| `_quality` | Llogarit cilesine e nje segmenti per te gjetur pjeset me te dobeta |
| `_can_add` | Kontrollon nese nje program mund te shtohet pa shkelur kufizimet |
| `_pick_candidate` | Zgjedh kandidatin me te mire ose nje kandidat nga top-k gjate repair |
| `_fill_interval` | Mbush intervalet boshe me programe te vlefshme |
| `_repair` | Rinderton orarin dhe ruan validitetin |
| `_weak_indexes` | Gjen segmentet me te dobeta te orarit |
| `_local_search` | Provon largimin e segmenteve te dobeta dhe rindertimin me repair |
| `_perturb` | Ben ndryshim te kontrolluar mbi zgjidhjen aktuale |
| `improve` | Ekzekuton ciklin kryesor te ILS |
| `run_ils_experiment` | Ekzekuton 10 runs per nje instance nga `main.py` |

### Operatoret e Perdorur

Ne kete faze jane perdorur operatore te thjeshte, te kuptueshem dhe te pershtatshem per scheduling:

| Operator | Qellimi |
|---|---|
| Remove weak segment | Largon nje segment me cilesi te ulet nga orari |
| Partial repair | Rinderton boshllikun me kandidat me score me te mire |
| Window perturbation | Largon nje numer te vogel segmentesh nga nje zone e orarit |
| Weak-list perturbation | Zgjedh disa segmente te dobeta dhe i zevendeson |
| Best-preserving acceptance | Pranon vetem zgjidhje qe permiresojne score-in |

Keta operatore u perdoren sepse nuk prishin krejt strukturen e seed-it te mire nga GA, por japin mundesi qe orari te permiresohet lokalisht.

### Parametrat e GA + ILS

Parametrat aktuale te fazes GA + ILS jane:

| Parameter | Value | Pershkrim |
|---|---:|---|
| `top_k` | 3 | Numri i kandidateve me te mire qe ruhen gjate repair |
| `random_repair` | 0.15 | Probabiliteti per te zgjedhur rastesisht nga top-k |
| `local_attempts` | 40 | Numri i tentimeve lokale brenda nje iterimi |
| `perturbation_size` | 4 | Numri maksimal i segmenteve qe mund te largohen gjate perturbimit |
| `max_stagnation` | 100 | Numri maksimal total i tentimeve pa permiresim |
| `time_per_run` | 300s | Maksimumi 5 minuta per cdo run |
| `candidate_pool_size` | 700 | Numri maksimal i kandidateve te ruajtur per repair |
| `runs` | 10 | Numri i ekzekutimeve per secilen instance |
| `base_experiment` | `exp_uniform_balanced` | Eksperimenti i GA nga i cili merret seed-i fillestar |

Algoritmi provon te gjeje permiresim, por nuk lejon qe output-i final te jete me i dobet se seed-i fillestar. Ne fund te cdo run ruhet maksimumi mes zgjidhjes se re dhe seed-it te GA. Nga `main.py`, cdo run ka limit maksimal 300 sekonda dhe nje instance ekzekutohet 10 here.

### Rezultatet e GA + ILS

Seed total nga zgjidhjet me te mira te `exp_uniform_balanced` eshte:

```text
167321
```

Total score i zgjidhjeve me te mira pas GA + ILS eshte:

```text
175826
```

Permiresimi ndaj seed-it te GA eshte:

```text
175826 - 167321 = +8505
```

Permiresimi ndaj Branch and Bound eshte:

```text
175826 - 136573 = +39253
```

Permbledhja e runs:

| Metric | Value |
|---|---:|
| Instances | 17 |
| Runs total | 170 |
| Improved runs | 54 |
| Same runs | 116 |
| Worse runs | 0 |
| Improved instances | 10 |
| Best total per run | 175601 |
| Best run | 5 |
| Sum of best instance scores | 175826 |

### Total Score per Secilin Run

| Run | GA + ILS total |
| ---: | ---: |
| 1 | 171263 |
| 2 | 175376 |
| 3 | 172680 |
| 4 | 174269 |
| 5 | 175601 |
| 6 | 171426 |
| 7 | 171506 |
| 8 | 171812 |
| 9 | 171237 |
| 10 | 170638 |

### Rezultatet e Plota te GA + ILS

| Instance | Seed GA | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Run 8 | Run 9 | Run 10 | Best | Avg | Worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| australia_iptv | 4456 | 4464 | 4459 | 4456 | 4465 | 4461 | 4464 | 4456 | 4459 | 4465 | 4464 | 4465 | 4461.3 | 4456 |
| canada_pw | 4822 | 4822 | 4822 | 4822 | 4822 | 4822 | 4824 | 4822 | 4822 | 4822 | 4822 | 4824 | 4822.2 | 4822 |
| china_pw | 2827 | 2858 | 2858 | 2858 | 2858 | 2858 | 2858 | 2858 | 2858 | 2858 | 2827 | 2858 | 2854.9 | 2827 |
| croatia_tv | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220 | 2220.0 | 2220 |
| france_iptv | 9730 | 9730 | 9730 | 9730 | 9730 | 9730 | 9730 | 9812 | 9730 | 9730 | 9751 | 9812 | 9740.3 | 9730 |
| germany_tv | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626 | 1626.0 | 1626 |
| kosovo_tv | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 2567 |
| netherlands_tv | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 2632 |
| singapore_pw | 4740 | 4767 | 4740 | 4767 | 4740 | 4767 | 4740 | 4783 | 4740 | 4767 | 4740 | 4783 | 4755.1 | 4740 |
| spain_iptv | 5501 | 5501 | 5597 | 5501 | 5587 | 5501 | 5612 | 5525 | 5502 | 5501 | 5560 | 5612 | 5538.7 | 5501 |
| toy | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510 | 510.0 | 510 |
| uk_iptv | 6408 | 6413 | 6408 | 6408 | 6408 | 6408 | 6408 | 6408 | 6408 | 6408 | 6413 | 6413 | 6409.0 | 6408 |
| uk_tv | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240 | 2240.0 | 2240 |
| us_iptv | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569 | 4569.0 | 4569 |
| usa_tv | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3575 | 3580 | 3580 | 3575.5 | 3575 |
| youtube_gold | 67576 | 67713 | 67667 | 67678 | 67685 | 67763 | 67732 | 67678 | 67723 | 67683 | 67686 | 67763 | 67700.8 | 67667 |
| youtube_premium | 41322 | 45056 | 49156 | 46521 | 48035 | 49352 | 45119 | 45225 | 45631 | 45064 | 44431 | 49352 | 46359.0 | 44431 |

### Interpretimi i Rezultateve te ILS

Rezultatet tregojne se ILS e permireson konfigurimin me te mire te algoritmit gjenetik pa prodhuar asnje run me score me te ulet se seed-i fillestar. Kjo ndodh sepse ne fund te cdo run ruhet maksimumi mes zgjidhjes se re dhe zgjidhjes fillestare.

Permiresimi me i madh ndodh te `youtube_premium`, ku score-i rritet nga `41322` ne `49352`. Kjo ndikon shume ne totalin final, sepse eshte instance me hapesire te madhe kerkimi dhe me shume mundesi per rindertim lokal.

Disa instanca nuk ndryshojne, sepse ose seed-i eshte tashme shume i forte, ose instanca ka hapesire te vogel kerkimi. Kjo shihet te `croatia_tv`, `germany_tv`, `kosovo_tv`, `netherlands_tv`, `toy`, `uk_tv` dhe `us_iptv`.

### Perfundimi i Faze se GA + ILS

Faza GA + ILS e rrit rezultatin final nga:

```text
166067
```

ne:

```text
175826
```

Kjo e ben GA + ILS fazen me te forte te projektit, sepse merr zgjidhjet me te mira te algoritmit gjenetik dhe i permireson edhe me tej me kerkimin lokal te iteruar.

## Si të Ekzekutohet Projekti

Së pari kalohet në folderin kryesor të implementimit:

```powershell
cd AA_25-26-main/AA_25-26-main
```

Për ekzekutim interaktiv përdoret:

```powershell
python main.py
```

Në këtë mënyrë shfaqet lista e instancave nga `data/input`, pastaj zgjedhet algoritmi:

```text
[1] Beam Search
[2] Branch and Bound
[3] Genetic Algorithm
[4] Iterated Local Search
```

Opsioni `1` ekzekuton Beam Search për instancën e zgjedhur dhe ruan rezultatin në:

```text
data/output/beam_search/<instance>
```

Opsioni `2` ekzekuton Branch and Bound për instancën e zgjedhur dhe ruan rezultatin në:

```text
data/output/branch_and_bound/<instance>
```

Opsioni `3` ekzekuton algoritmin gjenetik. Pas zgjedhjes së këtij opsioni, zgjidhet edhe njëri nga tri eksperimentet:

```text
[1] Parametra te njejte per te gjitha instancat
[2] Parametra te pershtatur sipas instances
[3] Parametra te pershtatur + eksplorim me i forte
```

Për ekzekutimin automatik të tri eksperimenteve të algoritmit gjenetik përdoret:

```powershell
python main.py --ga-auto --ga-runs 10 --ga-time 300
```

Për ekzekutim direkt të eksperimentit final të algoritmit gjenetik:

```powershell
python ga_experiment.py --runs 10 --total-time 5100 --max-instance-time 300 --profile single --experiment-name exp_uniform_balanced --clean-output
```

Opsioni `4` ekzekuton Iterated Local Search mbi output-in më të mirë ekzistues nga `exp_uniform_balanced`. Ky opsion nuk e ekzekuton prapë algoritmin gjenetik, por lexon seed-in ekzistues dhe tenton ta përmirësojë. Output-et ruhen në:

```text
data/output/ga_and_ils/<instance>
```
