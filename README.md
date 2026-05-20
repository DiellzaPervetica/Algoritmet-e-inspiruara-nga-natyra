<table>
  <tr>
    <td width="150" align="center" valign="center">
      <img src="logo.png" width="120" alt="University Logo" />
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
- [Rezultatet e BnB + GA + ILS](#rezultatet-e-bnb--ga--ils)
- [Rezultatet e Plota të BnB + GA + ILS](#rezultatet-e-plota-të-bnb--ga--ils)
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

Në fazën finale, opsioni `4` e ekzekuton pipeline-in në rendin:

```text
Branch and Bound -> Genetic Algorithm -> ILS
```

Për çdo run ruhen rezultatet e të tri fazave: `bnb_score`, `ga_score`, `ils_score` dhe `final_score`. ILS e merr si pikënisje rezultatin e GA-së nga i njëjti run dhe rezultati final ruhet si vlera më e mirë pas GA + ILS.

Output-et e kësaj faze ruhen në:

```text
data/output/bnb_ga_ils/<instance>/summary.json
```

Limitet e ekzekutimit final:

| Parameter | Value |
|---|---:|
| Runs për instancë | 10 |
| Limit Branch and Bound për run | 30s |
| Limit Genetic Algorithm për run | 240s |
| Limit ILS për run | 60s |
| Maksimumi për run | 330s / 5.5 min |
| Maksimumi për instancë | 55 min |

### Rezultatet e BnB + GA + ILS

Përmbledhja është nxjerrë nga `summary.json` për secilën instancë.

| Metric | Value |
|---|---:|
| Instances | 17 |
| Runs total | 170 |
| Runs improved by ILS | 94 |
| Runs unchanged by ILS | 76 |
| Runs worsened by ILS | 0 |
| Sum of best BnB scores | 138245 |
| Sum of best GA scores | 149090 |
| Sum of best final scores | 175954 |
| Improvement vs GA | +26864 |
| Improvement vs BnB | +37709 |

| Instance | Runs | Best BnB | Best GA | Best ILS | Best Final | Avg Final | Total GA improvement | Total ILS improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| australia_iptv | 10 | 4182 | 4427 | 4556 | 4556 | 4421.3 | 3722 | 1547 |
| canada_pw | 10 | 4615 | 4775 | 4775 | 4775 | 4723.8 | 1514 | 96 |
| china_pw | 10 | 2598 | 2850 | 2872 | 2872 | 2800.5 | 2487 | 155 |
| croatia_tv | 10 | 2120 | 2220 | 2220 | 2220 | 2220.0 | 1000 | 0 |
| france_iptv | 10 | 4112 | 9525 | 10014 | 10014 | 9689.9 | 50611 | 5739 |
| germany_tv | 10 | 1633 | 1633 | 1633 | 1633 | 1633.0 | 0 | 0 |
| kosovo_tv | 10 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 0 | 0 |
| netherlands_tv | 10 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 0 | 0 |
| singapore_pw | 10 | 4297 | 4744 | 5311 | 5311 | 5135.4 | 2869 | 5676 |
| spain_iptv | 10 | 4501 | 5496 | 6048 | 6048 | 5980.8 | 8817 | 6276 |
| toy | 10 | 510 | 510 | 510 | 510 | 510.0 | 0 | 0 |
| uk_iptv | 10 | 4906 | 6709 | 7248 | 7248 | 6775.9 | 12086 | 7745 |
| uk_tv | 10 | 2171 | 2246 | 2255 | 2255 | 2245.4 | 720 | 24 |
| us_iptv | 10 | 4190 | 4518 | 4528 | 4528 | 4437.9 | 3542 | 147 |
| usa_tv | 10 | 3582 | 3582 | 3582 | 3582 | 3577.9 | 0 | 0 |
| youtube_gold | 10 | 66988 | 67021 | 67444 | 67444 | 67342.4 | 962 | 3454 |
| youtube_premium | 10 | 22641 | 23635 | 47759 | 47759 | 46827.6 | 9420 | 234169 |

### Total Score për Secilin Run

| Run | BnB total | GA total | ILS total | Final total | GA improvement | ILS improvement |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 137349 | 147499 | 174743 | 174743 | 10150 | 27244 |
| 2 | 136853 | 146249 | 173294 | 173294 | 9396 | 27045 |
| 3 | 137017 | 146269 | 172893 | 172893 | 9252 | 26624 |
| 4 | 137618 | 148059 | 173802 | 173802 | 10441 | 25743 |
| 5 | 137360 | 147042 | 173050 | 173050 | 9682 | 26008 |
| 6 | 137000 | 146908 | 174470 | 174470 | 9908 | 27562 |
| 7 | 137332 | 146711 | 174459 | 174459 | 9379 | 27748 |
| 8 | 137207 | 146401 | 172672 | 172672 | 9194 | 26271 |
| 9 | 137398 | 147754 | 172865 | 172865 | 10356 | 25111 |
| 10 | 137296 | 147288 | 172960 | 172960 | 9992 | 25672 |

### Rezultatet e Plota të BnB + GA + ILS

Tabela në vijim paraqet secilin run për secilën instancë, me score për BnB, GA dhe ILS.

| Instance | Run | BnB | GA | GA improvement | ILS | ILS improvement | Final |
|---|---:|---:|---:|---:|---:|---:|---:|
| australia_iptv | 1 | 4182 | 4313 | 131 | 4556 | 243 | 4556 |
| australia_iptv | 2 | 3506 | 4167 | 661 | 4330 | 163 | 4330 |
| australia_iptv | 3 | 3703 | 4319 | 616 | 4473 | 154 | 4473 |
| australia_iptv | 4 | 4123 | 4200 | 77 | 4459 | 259 | 4459 |
| australia_iptv | 5 | 4181 | 4261 | 80 | 4465 | 204 | 4465 |
| australia_iptv | 6 | 3618 | 4186 | 568 | 4334 | 148 | 4334 |
| australia_iptv | 7 | 3693 | 4227 | 534 | 4479 | 252 | 4479 |
| australia_iptv | 8 | 3645 | 4255 | 610 | 4298 | 43 | 4298 |
| australia_iptv | 9 | 4172 | 4427 | 255 | 4431 | 4 | 4431 |
| australia_iptv | 10 | 4121 | 4311 | 190 | 4388 | 77 | 4388 |
| canada_pw | 1 | 4526 | 4678 | 152 | 4692 | 14 | 4692 |
| canada_pw | 2 | 4568 | 4698 | 130 | 4698 | 0 | 4698 |
| canada_pw | 3 | 4536 | 4670 | 134 | 4691 | 21 | 4691 |
| canada_pw | 4 | 4597 | 4684 | 87 | 4687 | 3 | 4687 |
| canada_pw | 5 | 4585 | 4775 | 190 | 4775 | 0 | 4775 |
| canada_pw | 6 | 4554 | 4679 | 125 | 4700 | 21 | 4700 |
| canada_pw | 7 | 4615 | 4737 | 122 | 4737 | 0 | 4737 |
| canada_pw | 8 | 4590 | 4737 | 147 | 4745 | 8 | 4745 |
| canada_pw | 9 | 4531 | 4739 | 208 | 4757 | 18 | 4757 |
| canada_pw | 10 | 4526 | 4745 | 219 | 4756 | 11 | 4756 |
| china_pw | 1 | 2480 | 2799 | 319 | 2806 | 7 | 2806 |
| china_pw | 2 | 2484 | 2683 | 199 | 2683 | 0 | 2683 |
| china_pw | 3 | 2548 | 2818 | 270 | 2840 | 22 | 2840 |
| china_pw | 4 | 2598 | 2822 | 224 | 2839 | 17 | 2839 |
| china_pw | 5 | 2514 | 2750 | 236 | 2778 | 28 | 2778 |
| china_pw | 6 | 2541 | 2831 | 290 | 2831 | 0 | 2831 |
| china_pw | 7 | 2522 | 2762 | 240 | 2762 | 0 | 2762 |
| china_pw | 8 | 2536 | 2712 | 176 | 2736 | 24 | 2736 |
| china_pw | 9 | 2553 | 2823 | 270 | 2858 | 35 | 2858 |
| china_pw | 10 | 2587 | 2850 | 263 | 2872 | 22 | 2872 |
| croatia_tv | 1 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 2 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 3 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 4 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 5 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 6 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 7 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 8 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 9 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| croatia_tv | 10 | 2120 | 2220 | 100 | 2220 | 0 | 2220 |
| france_iptv | 1 | 4112 | 8945 | 4833 | 9650 | 705 | 9650 |
| france_iptv | 2 | 4060 | 8915 | 4855 | 9630 | 715 | 9630 |
| france_iptv | 3 | 4085 | 8572 | 4487 | 9328 | 756 | 9328 |
| france_iptv | 4 | 4035 | 9525 | 5490 | 9962 | 437 | 9962 |
| france_iptv | 5 | 4022 | 9371 | 5349 | 9828 | 457 | 9828 |
| france_iptv | 6 | 4021 | 9281 | 5260 | 10014 | 733 | 10014 |
| france_iptv | 7 | 4065 | 8647 | 4582 | 9647 | 1000 | 9647 |
| france_iptv | 8 | 4018 | 9205 | 5187 | 9601 | 396 | 9601 |
| france_iptv | 9 | 4062 | 9289 | 5227 | 9415 | 126 | 9415 |
| france_iptv | 10 | 4069 | 9410 | 5341 | 9824 | 414 | 9824 |
| germany_tv | 1 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 2 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 3 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 4 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 5 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 6 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 7 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 8 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 9 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| germany_tv | 10 | 1633 | 1633 | 0 | 1633 | 0 | 1633 |
| kosovo_tv | 1 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 2 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 3 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 4 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 5 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 6 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 7 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 8 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 9 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| kosovo_tv | 10 | 2567 | 2567 | 0 | 2567 | 0 | 2567 |
| netherlands_tv | 1 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 2 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 3 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 4 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 5 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 6 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 7 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 8 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 9 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| netherlands_tv | 10 | 2632 | 2632 | 0 | 2632 | 0 | 2632 |
| singapore_pw | 1 | 4268 | 4554 | 286 | 5280 | 726 | 5280 |
| singapore_pw | 2 | 4268 | 4466 | 198 | 5060 | 594 | 5060 |
| singapore_pw | 3 | 4286 | 4527 | 241 | 5154 | 627 | 5154 |
| singapore_pw | 4 | 4297 | 4736 | 439 | 5221 | 485 | 5221 |
| singapore_pw | 5 | 4292 | 4629 | 337 | 5051 | 422 | 5051 |
| singapore_pw | 6 | 4282 | 4520 | 238 | 5311 | 791 | 5311 |
| singapore_pw | 7 | 4286 | 4615 | 329 | 5171 | 556 | 5171 |
| singapore_pw | 8 | 4286 | 4426 | 140 | 5258 | 832 | 5258 |
| singapore_pw | 9 | 4268 | 4461 | 193 | 4850 | 389 | 4850 |
| singapore_pw | 10 | 4276 | 4744 | 468 | 4998 | 254 | 4998 |
| spain_iptv | 1 | 4445 | 5379 | 934 | 6020 | 641 | 6020 |
| spain_iptv | 2 | 4445 | 5458 | 1013 | 5969 | 511 | 5969 |
| spain_iptv | 3 | 4445 | 5254 | 809 | 6016 | 762 | 6016 |
| spain_iptv | 4 | 4494 | 5394 | 900 | 5952 | 558 | 5952 |
| spain_iptv | 5 | 4444 | 5496 | 1052 | 6024 | 528 | 6024 |
| spain_iptv | 6 | 4501 | 5293 | 792 | 5949 | 656 | 5949 |
| spain_iptv | 7 | 4452 | 5334 | 882 | 5934 | 600 | 5934 |
| spain_iptv | 8 | 4501 | 5228 | 727 | 5910 | 682 | 5910 |
| spain_iptv | 9 | 4494 | 5316 | 822 | 5986 | 670 | 5986 |
| spain_iptv | 10 | 4494 | 5380 | 886 | 6048 | 668 | 6048 |
| toy | 1 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 2 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 3 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 4 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 5 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 6 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 7 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 8 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 9 | 510 | 510 | 0 | 510 | 0 | 510 |
| toy | 10 | 510 | 510 | 0 | 510 | 0 | 510 |
| uk_iptv | 1 | 4472 | 6709 | 2237 | 7248 | 539 | 7248 |
| uk_iptv | 2 | 4864 | 5549 | 685 | 6424 | 875 | 6424 |
| uk_iptv | 3 | 4837 | 5773 | 936 | 6668 | 895 | 6668 |
| uk_iptv | 4 | 4754 | 6429 | 1675 | 7112 | 683 | 7112 |
| uk_iptv | 5 | 4830 | 5753 | 923 | 6320 | 567 | 6320 |
| uk_iptv | 6 | 4738 | 5701 | 963 | 6975 | 1274 | 6975 |
| uk_iptv | 7 | 4835 | 6246 | 1411 | 6919 | 673 | 6919 |
| uk_iptv | 8 | 4906 | 5781 | 875 | 6255 | 474 | 6255 |
| uk_iptv | 9 | 4855 | 6395 | 1540 | 6935 | 540 | 6935 |
| uk_iptv | 10 | 4837 | 5678 | 841 | 6903 | 1225 | 6903 |
| uk_tv | 1 | 2171 | 2246 | 75 | 2255 | 9 | 2255 |
| uk_tv | 2 | 2171 | 2240 | 69 | 2240 | 0 | 2240 |
| uk_tv | 3 | 2171 | 2240 | 69 | 2240 | 0 | 2240 |
| uk_tv | 4 | 2171 | 2240 | 69 | 2240 | 0 | 2240 |
| uk_tv | 5 | 2171 | 2246 | 75 | 2255 | 9 | 2255 |
| uk_tv | 6 | 2171 | 2240 | 69 | 2240 | 0 | 2240 |
| uk_tv | 7 | 2171 | 2240 | 69 | 2246 | 6 | 2246 |
| uk_tv | 8 | 2171 | 2246 | 75 | 2246 | 0 | 2246 |
| uk_tv | 9 | 2171 | 2246 | 75 | 2246 | 0 | 2246 |
| uk_tv | 10 | 2171 | 2246 | 75 | 2246 | 0 | 2246 |
| us_iptv | 1 | 4096 | 4421 | 325 | 4431 | 10 | 4431 |
| us_iptv | 2 | 3885 | 4333 | 448 | 4335 | 2 | 4335 |
| us_iptv | 3 | 4120 | 4379 | 259 | 4393 | 14 | 4393 |
| us_iptv | 4 | 4190 | 4373 | 183 | 4464 | 91 | 4464 |
| us_iptv | 5 | 4008 | 4483 | 475 | 4489 | 6 | 4489 |
| us_iptv | 6 | 4168 | 4518 | 350 | 4518 | 0 | 4518 |
| us_iptv | 7 | 4108 | 4348 | 240 | 4354 | 6 | 4354 |
| us_iptv | 8 | 4112 | 4510 | 398 | 4528 | 18 | 4528 |
| us_iptv | 9 | 4005 | 4455 | 450 | 4455 | 0 | 4455 |
| us_iptv | 10 | 3998 | 4412 | 414 | 4412 | 0 | 4412 |
| usa_tv | 1 | 3575 | 3575 | 0 | 3575 | 0 | 3575 |
| usa_tv | 2 | 3579 | 3579 | 0 | 3579 | 0 | 3579 |
| usa_tv | 3 | 3575 | 3575 | 0 | 3575 | 0 | 3575 |
| usa_tv | 4 | 3575 | 3575 | 0 | 3575 | 0 | 3575 |
| usa_tv | 5 | 3582 | 3582 | 0 | 3582 | 0 | 3582 |
| usa_tv | 6 | 3582 | 3582 | 0 | 3582 | 0 | 3582 |
| usa_tv | 7 | 3575 | 3575 | 0 | 3575 | 0 | 3575 |
| usa_tv | 8 | 3575 | 3575 | 0 | 3575 | 0 | 3575 |
| usa_tv | 9 | 3579 | 3579 | 0 | 3579 | 0 | 3579 |
| usa_tv | 10 | 3582 | 3582 | 0 | 3582 | 0 | 3582 |
| youtube_gold | 1 | 66919 | 67021 | 102 | 67444 | 423 | 67444 |
| youtube_gold | 2 | 66960 | 66964 | 4 | 67351 | 387 | 67351 |
| youtube_gold | 3 | 66909 | 67001 | 92 | 67280 | 279 | 67280 |
| youtube_gold | 4 | 66970 | 66984 | 14 | 67322 | 338 | 67322 |
| youtube_gold | 5 | 66889 | 66984 | 95 | 67278 | 294 | 67278 |
| youtube_gold | 6 | 66919 | 66993 | 74 | 67361 | 368 | 67361 |
| youtube_gold | 7 | 66988 | 67001 | 13 | 67314 | 313 | 67314 |
| youtube_gold | 8 | 66803 | 67021 | 218 | 67309 | 288 | 67309 |
| youtube_gold | 9 | 66825 | 67000 | 175 | 67339 | 339 | 67339 |
| youtube_gold | 10 | 66826 | 67001 | 175 | 67426 | 425 | 67426 |
| youtube_premium | 1 | 22641 | 23297 | 656 | 47224 | 23927 | 47224 |
| youtube_premium | 2 | 22601 | 23635 | 1034 | 47433 | 23798 | 47433 |
| youtube_premium | 3 | 22340 | 23579 | 1239 | 46673 | 23094 | 46673 |
| youtube_premium | 4 | 22352 | 23535 | 1183 | 46407 | 22872 | 46407 |
| youtube_premium | 5 | 22380 | 23150 | 770 | 46643 | 23493 | 46643 |
| youtube_premium | 6 | 22443 | 23522 | 1079 | 47093 | 23571 | 47093 |
| youtube_premium | 7 | 22560 | 23417 | 857 | 47759 | 24342 | 47759 |
| youtube_premium | 8 | 22602 | 23143 | 541 | 46649 | 23506 | 46649 |
| youtube_premium | 9 | 22421 | 23462 | 1041 | 46452 | 22990 | 46452 |
| youtube_premium | 10 | 22347 | 23367 | 1020 | 45943 | 22576 | 45943 |

## Si të Ekzekutohet Projekti

Së pari kalohet në folderin kryesor të implementimit:

```powershell
cd AA_25-26-main/AA_25-26-main
```

Për ekzekutim interaktiv përdoret:

```powershell
python main.py
```

Pas kësaj shfaqet lista e instancave nga `data/input`, pastaj zgjedhet algoritmi:

```text
[1] Beam Search
[2] Branch and Bound
[3] Genetic Algorithm
[4] Branch and Bound + Genetic Algorithm + ILS
```

Opsioni `4` ekzekuton pipeline-in final `Branch and Bound -> Genetic Algorithm -> ILS` për instancën e zgjedhur. Për çdo instance bëhen `10` runs, ndërsa për çdo run ruhen score-t në `summary.json`.

Output-et ruhen në:

```text
data/output/bnb_ga_ils/<instance>
```

Për ekzekutim me parametrat finalë nga terminali mund të përdoret:

```powershell
python main.py --pipeline-runs 10 --bnb-time 30 --pipeline-ga-time 240 --ils-time 60 --pipeline-ga-profile single
```
