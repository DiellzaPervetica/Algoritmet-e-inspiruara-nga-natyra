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
- [Rezultatet e Plota te BnB + GA + ILS](#rezultatet-e-plota-te-bnb--ga--ils)
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

Për çdo run ruhen rezultatet e të tri fazave: `bnb_score`, `ga_score`, `ils_score`, `final_score`, si edhe koha e secilës fazë. ILS e merr si pikënisje rezultatin e GA-së nga i njëjti run dhe rezultati final ruhet si vlera më e mirë pas GA + ILS.

Output-et e kësaj faze ruhen në:

```text
data/output/bnb_ga_ils/<instance>/summary.json
```

Parametrat e ekzekutimit final:

| Parameter | Value |
|---|---:|
| Runs për instancë | 10 |
| Branch and Bound për run | 30s |
| Genetic Algorithm për run | 240s |
| ILS për run | 60s |
| Maksimumi për run | 330s / 5.5 min |
| Maksimumi për instancë | 55 min |

### Rezultatet e BnB + GA + ILS

Përmbledhja është nxjerrë nga `summary.json` për secilën instancë.

| Metric | Value |
|---|---:|
| Instances | 17 |
| Runs total | 170 |
| Improved runs | 94 |
| Same runs | 76 |
| Worse runs | 0 |
| Sum of best BnB scores | 138224 |
| Sum of best GA scores | 149083 |
| Sum of best final scores | 175947 |
| Improvement vs GA | +26864 |
| Improvement vs BnB | +37723 |
| Total recorded time | 6.57 h |

| Instance | Runs | Best BnB | Best GA | Best ILS | Best Final | Avg Final | Total improvement | Total time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| australia_iptv | 10 | 4182 | 4427 | 4556 | 4556 | 4421.3 | 1547 | 20.28 |
| canada_pw | 10 | 4615 | 4775 | 4775 | 4775 | 4723.8 | 96 | 18.64 |
| china_pw | 10 | 2598 | 2850 | 2872 | 2872 | 2800.5 | 155 | 21.63 |
| croatia_tv | 10 | 2120 | 2220 | 2220 | 2220 | 2220.0 | 0 | 18.34 |
| france_iptv | 10 | 4112 | 9525 | 10014 | 10014 | 9689.9 | 5739 | 27.92 |
| germany_tv | 10 | 1633 | 1633 | 1633 | 1633 | 1633.0 | 0 | 18.35 |
| kosovo_tv | 10 | 2567 | 2567 | 2567 | 2567 | 2567.0 | 0 | 18.35 |
| netherlands_tv | 10 | 2632 | 2632 | 2632 | 2632 | 2632.0 | 0 | 18.35 |
| singapore_pw | 10 | 4297 | 4744 | 5311 | 5311 | 5135.4 | 5676 | 18.53 |
| spain_iptv | 10 | 4501 | 5496 | 6048 | 6048 | 5980.8 | 6276 | 19.08 |
| toy | 10 | 510 | 510 | 510 | 510 | 510.0 | 0 | 13.34 |
| uk_iptv | 10 | 4906 | 6709 | 7248 | 7248 | 6775.9 | 7745 | 23.51 |
| uk_tv | 10 | 2171 | 2246 | 2255 | 2255 | 2245.4 | 24 | 18.35 |
| us_iptv | 10 | 4190 | 4518 | 4528 | 4528 | 4437.9 | 147 | 121.66 |
| usa_tv | 10 | 3561 | 3575 | 3575 | 3575 | 3575.0 | 0 | 5.86 |
| youtube_gold | 10 | 66988 | 67021 | 67444 | 67444 | 67342.4 | 3454 | 5.91 |
| youtube_premium | 10 | 22641 | 23635 | 47759 | 47759 | 46827.6 | 234169 | 5.89 |

### Total Score per Secilin Run

| Run | BnB total | GA total | ILS total | Final total | Improvement | Time (min) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 137335 | 147499 | 174743 | 174743 | 27244 | 50.47 |
| 2 | 136835 | 146245 | 173290 | 173290 | 27045 | 50.78 |
| 3 | 137003 | 146269 | 172893 | 172893 | 26624 | 50.44 |
| 4 | 137604 | 148059 | 173802 | 173802 | 25743 | 51.90 |
| 5 | 137339 | 147035 | 173043 | 173043 | 26008 | 50.94 |
| 6 | 136979 | 146901 | 174463 | 174463 | 27562 | 27.90 |
| 7 | 137318 | 146711 | 174459 | 174459 | 27748 | 27.70 |
| 8 | 137193 | 146401 | 172672 | 172672 | 26271 | 28.16 |
| 9 | 137380 | 147750 | 172861 | 172861 | 25111 | 27.93 |
| 10 | 137275 | 147281 | 172953 | 172953 | 25672 | 27.75 |

### Rezultatet e Plota te BnB + GA + ILS

Tabela në vijim paraqet secilin run për secilën instancë, me score dhe kohë për BnB, GA dhe ILS.

| Instance | Run | Seed | BnB | GA | ILS | Final | Improvement | BnB s | GA s | ILS s | Total s | GA gen | ILS iter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| australia_iptv | 1 | 670488 | 4182 | 4313 | 4556 | 4556 | 243 | 30.31 | 24.27 | 60.05 | 114.64 | 38 | 5576 |
| australia_iptv | 2 | 116740 | 3506 | 4167 | 4330 | 4330 | 163 | 30.54 | 25.60 | 60.05 | 116.18 | 43 | 7475 |
| australia_iptv | 3 | 26226 | 3703 | 4319 | 4473 | 4473 | 154 | 30.76 | 30.85 | 60.05 | 121.66 | 48 | 6282 |
| australia_iptv | 4 | 777573 | 4123 | 4200 | 4459 | 4459 | 259 | 30.28 | 23.53 | 60.04 | 113.85 | 35 | 7796 |
| australia_iptv | 5 | 288390 | 4181 | 4261 | 4465 | 4465 | 204 | 30.26 | 34.74 | 60.04 | 125.04 | 48 | 6731 |
| australia_iptv | 6 | 256788 | 3618 | 4186 | 4334 | 4334 | 148 | 30.80 | 32.44 | 60.05 | 123.30 | 25 | 5265 |
| australia_iptv | 7 | 234054 | 3693 | 4227 | 4479 | 4479 | 252 | 30.26 | 32.77 | 60.05 | 123.08 | 34 | 5237 |
| australia_iptv | 8 | 146317 | 3645 | 4255 | 4298 | 4298 | 43 | 30.37 | 48.41 | 60.03 | 138.81 | 66 | 6494 |
| australia_iptv | 9 | 772247 | 4172 | 4427 | 4431 | 4431 | 4 | 30.35 | 28.96 | 60.05 | 119.36 | 42 | 7445 |
| australia_iptv | 10 | 107474 | 4121 | 4311 | 4388 | 4388 | 77 | 30.35 | 30.19 | 60.03 | 120.58 | 44 | 7626 |
| canada_pw | 1 | 670488 | 4526 | 4678 | 4692 | 4692 | 14 | 30.47 | 20.10 | 60.05 | 110.61 | 30 | 6336 |
| canada_pw | 2 | 116740 | 4568 | 4698 | 4698 | 4698 | 0 | 30.42 | 20.35 | 60.07 | 110.83 | 35 | 6672 |
| canada_pw | 3 | 26226 | 4536 | 4670 | 4691 | 4691 | 21 | 30.36 | 20.77 | 60.11 | 111.24 | 26 | 5415 |
| canada_pw | 4 | 777573 | 4597 | 4684 | 4687 | 4687 | 3 | 30.37 | 20.96 | 60.05 | 111.39 | 33 | 5912 |
| canada_pw | 5 | 288390 | 4585 | 4775 | 4775 | 4775 | 0 | 30.49 | 20.29 | 60.06 | 110.83 | 36 | 7331 |
| canada_pw | 6 | 256788 | 4554 | 4679 | 4700 | 4700 | 21 | 30.42 | 20.08 | 60.05 | 110.55 | 31 | 6397 |
| canada_pw | 7 | 234054 | 4615 | 4737 | 4737 | 4737 | 0 | 30.28 | 20.79 | 60.05 | 111.12 | 27 | 3420 |
| canada_pw | 8 | 146317 | 4590 | 4737 | 4745 | 4745 | 8 | 30.43 | 21.29 | 60.08 | 111.80 | 20 | 3657 |
| canada_pw | 9 | 772247 | 4531 | 4739 | 4757 | 4757 | 18 | 30.31 | 21.50 | 60.06 | 111.87 | 39 | 6823 |
| canada_pw | 10 | 107474 | 4526 | 4745 | 4756 | 4756 | 11 | 30.40 | 27.42 | 60.05 | 117.86 | 44 | 7005 |
| china_pw | 1 | 670488 | 2480 | 2799 | 2806 | 2806 | 7 | 41.46 | 25.70 | 60.08 | 127.24 | 36 | 4684 |
| china_pw | 2 | 116740 | 2484 | 2683 | 2683 | 2683 | 0 | 44.78 | 20.35 | 60.07 | 125.19 | 37 | 4541 |
| china_pw | 3 | 26226 | 2548 | 2818 | 2840 | 2840 | 22 | 39.78 | 32.87 | 60.06 | 132.71 | 50 | 4403 |
| china_pw | 4 | 777573 | 2598 | 2822 | 2839 | 2839 | 17 | 39.81 | 24.30 | 60.12 | 124.23 | 38 | 4361 |
| china_pw | 5 | 288390 | 2514 | 2750 | 2778 | 2778 | 28 | 40.26 | 27.80 | 60.07 | 128.13 | 50 | 4270 |
| china_pw | 6 | 256788 | 2541 | 2831 | 2831 | 2831 | 0 | 40.42 | 24.14 | 60.06 | 124.62 | 44 | 5147 |
| china_pw | 7 | 234054 | 2522 | 2762 | 2762 | 2762 | 0 | 52.20 | 25.11 | 60.17 | 137.48 | 25 | 2901 |
| china_pw | 8 | 146317 | 2536 | 2712 | 2736 | 2736 | 24 | 59.15 | 20.84 | 60.08 | 140.07 | 35 | 4954 |
| china_pw | 9 | 772247 | 2553 | 2823 | 2858 | 2858 | 35 | 42.18 | 32.67 | 60.07 | 134.92 | 49 | 5405 |
| china_pw | 10 | 107474 | 2587 | 2850 | 2872 | 2872 | 22 | 40.02 | 23.29 | 60.07 | 123.37 | 41 | 4837 |
| croatia_tv | 1 | 670488 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.08 | 60.01 | 110.10 | 160 | 113836 |
| croatia_tv | 2 | 116740 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.04 | 60.01 | 110.05 | 142 | 100460 |
| croatia_tv | 3 | 26226 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.09 | 60.01 | 110.10 | 161 | 110623 |
| croatia_tv | 4 | 777573 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.08 | 60.01 | 110.09 | 157 | 114938 |
| croatia_tv | 5 | 288390 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.02 | 60.01 | 110.04 | 128 | 85695 |
| croatia_tv | 6 | 256788 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.02 | 60.01 | 110.04 | 132 | 115606 |
| croatia_tv | 7 | 234054 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.02 | 60.01 | 110.03 | 153 | 96039 |
| croatia_tv | 8 | 146317 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.05 | 60.01 | 110.06 | 140 | 113448 |
| croatia_tv | 9 | 772247 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.02 | 60.01 | 110.03 | 155 | 110168 |
| croatia_tv | 10 | 107474 | 2120 | 2220 | 2220 | 2220 | 0 | 30.00 | 20.07 | 60.01 | 110.08 | 163 | 117688 |
| france_iptv | 1 | 670488 | 4112 | 8945 | 9650 | 9650 | 705 | 30.33 | 72.60 | 60.05 | 162.98 | 135 | 18282 |
| france_iptv | 2 | 116740 | 4060 | 8915 | 9630 | 9630 | 715 | 30.50 | 90.92 | 60.05 | 181.48 | 162 | 18804 |
| france_iptv | 3 | 26226 | 4085 | 8572 | 9328 | 9328 | 756 | 30.54 | 40.16 | 60.05 | 130.75 | 79 | 19341 |
| france_iptv | 4 | 777573 | 4035 | 9525 | 9962 | 9962 | 437 | 30.12 | 97.38 | 60.05 | 187.55 | 161 | 18512 |
| france_iptv | 5 | 288390 | 4022 | 9371 | 9828 | 9828 | 457 | 30.14 | 72.63 | 60.04 | 162.81 | 124 | 18338 |
| france_iptv | 6 | 256788 | 4021 | 9281 | 10014 | 10014 | 733 | 30.40 | 93.86 | 60.05 | 184.32 | 161 | 18678 |
| france_iptv | 7 | 234054 | 4065 | 8647 | 9647 | 9647 | 1000 | 30.46 | 61.48 | 60.05 | 151.98 | 116 | 19855 |
| france_iptv | 8 | 146317 | 4018 | 9205 | 9601 | 9601 | 396 | 30.23 | 77.57 | 60.05 | 167.85 | 142 | 19642 |
| france_iptv | 9 | 772247 | 4062 | 9289 | 9415 | 9415 | 126 | 30.15 | 82.01 | 60.05 | 172.21 | 145 | 19291 |
| france_iptv | 10 | 107474 | 4069 | 9410 | 9824 | 9824 | 414 | 30.28 | 82.71 | 60.06 | 173.04 | 147 | 19182 |
| germany_tv | 1 | 670488 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.03 | 60.02 | 110.05 | 189 | 157129 |
| germany_tv | 2 | 116740 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.05 | 60.02 | 110.07 | 184 | 156547 |
| germany_tv | 3 | 26226 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.11 | 60.02 | 110.13 | 193 | 155822 |
| germany_tv | 4 | 777573 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.09 | 60.02 | 110.11 | 189 | 156284 |
| germany_tv | 5 | 288390 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.02 | 60.02 | 110.04 | 178 | 156224 |
| germany_tv | 6 | 256788 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.01 | 60.02 | 110.03 | 185 | 154489 |
| germany_tv | 7 | 234054 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.07 | 60.02 | 110.09 | 182 | 156393 |
| germany_tv | 8 | 146317 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.01 | 60.02 | 110.03 | 185 | 155636 |
| germany_tv | 9 | 772247 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.12 | 60.02 | 110.14 | 153 | 152334 |
| germany_tv | 10 | 107474 | 1633 | 1633 | 1633 | 1633 | 0 | 30.00 | 20.08 | 60.02 | 110.10 | 192 | 157340 |
| kosovo_tv | 1 | 670488 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.04 | 60.02 | 110.06 | 152 | 118969 |
| kosovo_tv | 2 | 116740 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.05 | 60.02 | 110.07 | 159 | 120998 |
| kosovo_tv | 3 | 26226 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.07 | 60.02 | 110.10 | 160 | 119922 |
| kosovo_tv | 4 | 777573 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.13 | 60.03 | 110.16 | 155 | 120973 |
| kosovo_tv | 5 | 288390 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.02 | 60.02 | 110.05 | 153 | 114629 |
| kosovo_tv | 6 | 256788 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.04 | 60.02 | 110.06 | 158 | 120438 |
| kosovo_tv | 7 | 234054 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.13 | 60.02 | 110.15 | 154 | 121423 |
| kosovo_tv | 8 | 146317 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.04 | 60.02 | 110.07 | 156 | 118364 |
| kosovo_tv | 9 | 772247 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.13 | 60.02 | 110.16 | 142 | 118941 |
| kosovo_tv | 10 | 107474 | 2567 | 2567 | 2567 | 2567 | 0 | 30.00 | 20.03 | 60.03 | 110.06 | 152 | 120188 |
| netherlands_tv | 1 | 670488 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.11 | 60.02 | 110.14 | 142 | 111371 |
| netherlands_tv | 2 | 116740 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.09 | 60.02 | 110.11 | 142 | 110167 |
| netherlands_tv | 3 | 26226 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.10 | 60.02 | 110.12 | 139 | 110638 |
| netherlands_tv | 4 | 777573 | 2632 | 2632 | 2632 | 2632 | 0 | 30.01 | 20.09 | 60.03 | 110.13 | 113 | 82722 |
| netherlands_tv | 5 | 288390 | 2632 | 2632 | 2632 | 2632 | 0 | 30.01 | 20.11 | 60.02 | 110.13 | 45 | 37828 |
| netherlands_tv | 6 | 256788 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.07 | 60.02 | 110.10 | 132 | 110875 |
| netherlands_tv | 7 | 234054 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.09 | 60.02 | 110.11 | 139 | 112911 |
| netherlands_tv | 8 | 146317 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.11 | 60.02 | 110.13 | 141 | 112251 |
| netherlands_tv | 9 | 772247 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.08 | 60.02 | 110.10 | 133 | 109948 |
| netherlands_tv | 10 | 107474 | 2632 | 2632 | 2632 | 2632 | 0 | 30.00 | 20.03 | 60.02 | 110.06 | 139 | 113409 |
| singapore_pw | 1 | 670488 | 4268 | 4554 | 5280 | 5280 | 726 | 30.11 | 21.98 | 60.03 | 112.11 | 85 | 32809 |
| singapore_pw | 2 | 116740 | 4268 | 4466 | 5060 | 5060 | 594 | 30.10 | 20.21 | 60.03 | 110.34 | 68 | 31337 |
| singapore_pw | 3 | 26226 | 4286 | 4527 | 5154 | 5154 | 627 | 30.11 | 21.28 | 60.03 | 111.42 | 74 | 31914 |
| singapore_pw | 4 | 777573 | 4297 | 4736 | 5221 | 5221 | 485 | 30.08 | 25.24 | 60.04 | 115.35 | 94 | 30705 |
| singapore_pw | 5 | 288390 | 4292 | 4629 | 5051 | 5051 | 422 | 30.11 | 20.45 | 60.03 | 110.59 | 74 | 33090 |
| singapore_pw | 6 | 256788 | 4282 | 4520 | 5311 | 5311 | 791 | 30.07 | 20.27 | 60.04 | 110.37 | 73 | 32636 |
| singapore_pw | 7 | 234054 | 4286 | 4615 | 5171 | 5171 | 556 | 30.07 | 20.30 | 60.03 | 110.40 | 76 | 30585 |
| singapore_pw | 8 | 146317 | 4286 | 4426 | 5258 | 5258 | 832 | 30.09 | 20.11 | 60.03 | 110.23 | 71 | 29652 |
| singapore_pw | 9 | 772247 | 4268 | 4461 | 4850 | 4850 | 389 | 30.10 | 20.12 | 60.03 | 110.25 | 68 | 28950 |
| singapore_pw | 10 | 107474 | 4276 | 4744 | 4998 | 4998 | 254 | 30.09 | 20.48 | 60.03 | 110.61 | 75 | 31163 |
| spain_iptv | 1 | 670488 | 4445 | 5379 | 6020 | 6020 | 641 | 30.18 | 20.34 | 60.04 | 110.56 | 65 | 27548 |
| spain_iptv | 2 | 116740 | 4445 | 5458 | 5969 | 5969 | 511 | 30.19 | 20.15 | 60.04 | 110.38 | 65 | 28664 |
| spain_iptv | 3 | 26226 | 4445 | 5254 | 6016 | 6016 | 762 | 30.18 | 20.12 | 60.04 | 110.33 | 67 | 29299 |
| spain_iptv | 4 | 777573 | 4494 | 5394 | 5952 | 5952 | 558 | 30.20 | 30.63 | 60.03 | 120.86 | 104 | 30803 |
| spain_iptv | 5 | 288390 | 4444 | 5496 | 6024 | 6024 | 528 | 30.19 | 41.97 | 60.04 | 132.20 | 131 | 27723 |
| spain_iptv | 6 | 256788 | 4501 | 5293 | 5949 | 5949 | 656 | 30.20 | 20.10 | 60.04 | 110.33 | 68 | 28672 |
| spain_iptv | 7 | 234054 | 4452 | 5334 | 5934 | 5934 | 600 | 30.19 | 21.81 | 60.04 | 112.04 | 70 | 30485 |
| spain_iptv | 8 | 146317 | 4501 | 5228 | 5910 | 5910 | 682 | 30.18 | 20.31 | 60.04 | 110.53 | 66 | 29343 |
| spain_iptv | 9 | 772247 | 4494 | 5316 | 5986 | 5986 | 670 | 30.18 | 26.98 | 60.04 | 117.19 | 87 | 29298 |
| spain_iptv | 10 | 107474 | 4494 | 5380 | 6048 | 6048 | 668 | 30.20 | 20.10 | 60.04 | 110.34 | 64 | 28733 |
| toy | 1 | 670488 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.01 | 60.00 | 80.01 | 786 | 708472 |
| toy | 2 | 116740 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.02 | 60.00 | 80.02 | 771 | 705717 |
| toy | 3 | 26226 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.00 | 60.00 | 80.00 | 773 | 713454 |
| toy | 4 | 777573 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.02 | 60.00 | 80.03 | 739 | 685823 |
| toy | 5 | 288390 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.01 | 60.00 | 80.02 | 712 | 703040 |
| toy | 6 | 256788 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.01 | 60.00 | 80.01 | 728 | 695595 |
| toy | 7 | 234054 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.02 | 60.00 | 80.02 | 782 | 718046 |
| toy | 8 | 146317 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.01 | 60.00 | 80.01 | 771 | 681254 |
| toy | 9 | 772247 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.01 | 60.00 | 80.01 | 750 | 708489 |
| toy | 10 | 107474 | 510 | 510 | 510 | 510 | 0 | 0.00 | 20.00 | 60.00 | 80.00 | 776 | 680002 |
| uk_iptv | 1 | 670488 | 4472 | 6709 | 7248 | 7248 | 539 | 50.87 | 38.69 | 60.07 | 149.62 | 90 | 16564 |
| uk_iptv | 2 | 116740 | 4864 | 5549 | 6424 | 6424 | 875 | 49.82 | 21.02 | 60.06 | 130.90 | 49 | 16071 |
| uk_iptv | 3 | 26226 | 4837 | 5773 | 6668 | 6668 | 895 | 49.27 | 38.08 | 60.06 | 147.41 | 95 | 15013 |
| uk_iptv | 4 | 777573 | 4754 | 6429 | 7112 | 7112 | 683 | 50.53 | 34.48 | 60.08 | 145.09 | 83 | 15150 |
| uk_iptv | 5 | 288390 | 4830 | 5753 | 6320 | 6320 | 567 | 49.17 | 34.41 | 60.06 | 143.64 | 84 | 14335 |
| uk_iptv | 6 | 256788 | 4738 | 5701 | 6975 | 6975 | 1274 | 49.24 | 29.04 | 60.06 | 138.34 | 68 | 15176 |
| uk_iptv | 7 | 234054 | 4835 | 6246 | 6919 | 6919 | 673 | 48.89 | 34.13 | 60.06 | 143.08 | 83 | 15078 |
| uk_iptv | 8 | 146317 | 4906 | 5781 | 6255 | 6255 | 474 | 49.18 | 28.94 | 60.06 | 138.18 | 73 | 14331 |
| uk_iptv | 9 | 772247 | 4855 | 6395 | 6935 | 6935 | 540 | 49.37 | 27.86 | 60.07 | 137.30 | 65 | 17031 |
| uk_iptv | 10 | 107474 | 4837 | 5678 | 6903 | 6903 | 1225 | 50.47 | 26.43 | 60.06 | 136.96 | 67 | 13093 |
| uk_tv | 1 | 670488 | 2171 | 2246 | 2255 | 2255 | 9 | 30.03 | 20.08 | 60.01 | 110.12 | 110 | 50380 |
| uk_tv | 2 | 116740 | 2171 | 2240 | 2240 | 2240 | 0 | 30.02 | 20.10 | 60.01 | 110.13 | 121 | 52352 |
| uk_tv | 3 | 26226 | 2171 | 2240 | 2240 | 2240 | 0 | 30.02 | 20.03 | 60.01 | 110.06 | 126 | 54095 |
| uk_tv | 4 | 777573 | 2171 | 2240 | 2240 | 2240 | 0 | 30.01 | 20.11 | 60.01 | 110.14 | 127 | 54159 |
| uk_tv | 5 | 288390 | 2171 | 2246 | 2255 | 2255 | 9 | 30.02 | 20.03 | 60.01 | 110.05 | 117 | 54045 |
| uk_tv | 6 | 256788 | 2171 | 2240 | 2240 | 2240 | 0 | 30.02 | 20.05 | 60.01 | 110.08 | 131 | 54739 |
| uk_tv | 7 | 234054 | 2171 | 2240 | 2246 | 2246 | 6 | 30.02 | 20.14 | 60.02 | 110.17 | 129 | 53388 |
| uk_tv | 8 | 146317 | 2171 | 2246 | 2246 | 2246 | 0 | 30.02 | 20.10 | 60.01 | 110.13 | 123 | 53413 |
| uk_tv | 9 | 772247 | 2171 | 2246 | 2246 | 2246 | 0 | 30.02 | 20.07 | 60.01 | 110.10 | 123 | 53430 |
| uk_tv | 10 | 107474 | 2171 | 2246 | 2246 | 2246 | 0 | 30.02 | 20.17 | 60.01 | 110.20 | 129 | 53352 |
| us_iptv | 1 | 670488 | 4096 | 4421 | 4431 | 4431 | 10 | 1264.85 | 78.08 | 60.52 | 1403.45 | 48 | 1539 |
| us_iptv | 2 | 116740 | 3885 | 4333 | 4335 | 4335 | 2 | 1281.89 | 82.72 | 60.38 | 1424.99 | 49 | 1654 |
| us_iptv | 3 | 26226 | 4120 | 4379 | 4393 | 4393 | 14 | 1314.48 | 49.38 | 60.43 | 1424.29 | 30 | 1526 |
| us_iptv | 4 | 777573 | 4190 | 4373 | 4464 | 4464 | 91 | 1338.03 | 60.87 | 60.37 | 1459.27 | 38 | 1255 |
| us_iptv | 5 | 288390 | 4008 | 4483 | 4489 | 4489 | 6 | 1275.81 | 70.70 | 60.63 | 1407.14 | 47 | 1570 |
| us_iptv | 6 | 256788 | 4168 | 4518 | 4518 | 4518 | 0 | 0.00 | 20.44 | 15.41 | 35.88 | 14 | 374 |
| us_iptv | 7 | 234054 | 4108 | 4348 | 4354 | 4354 | 6 | 0.00 | 20.42 | 15.68 | 36.15 | 12 | 289 |
| us_iptv | 8 | 146317 | 4112 | 4510 | 4528 | 4528 | 18 | 0.00 | 20.54 | 15.40 | 35.96 | 12 | 414 |
| us_iptv | 9 | 772247 | 4005 | 4455 | 4455 | 4455 | 0 | 0.00 | 20.48 | 16.13 | 36.63 | 13 | 401 |
| us_iptv | 10 | 107474 | 3998 | 4412 | 4412 | 4412 | 0 | 0.00 | 20.50 | 15.40 | 35.91 | 14 | 382 |
| usa_tv | 1 | 670488 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.08 | 15.04 | 35.14 | 29 | 1130 |
| usa_tv | 2 | 116740 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.07 | 15.07 | 35.16 | 29 | 1126 |
| usa_tv | 3 | 26226 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.07 | 15.06 | 35.17 | 27 | 1131 |
| usa_tv | 4 | 777573 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.06 | 15.05 | 35.13 | 28 | 1285 |
| usa_tv | 5 | 288390 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.07 | 15.04 | 35.13 | 29 | 1351 |
| usa_tv | 6 | 256788 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.06 | 15.15 | 35.25 | 29 | 1203 |
| usa_tv | 7 | 234054 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.07 | 15.05 | 35.14 | 27 | 1284 |
| usa_tv | 8 | 146317 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.06 | 15.06 | 35.14 | 27 | 1240 |
| usa_tv | 9 | 772247 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.06 | 15.05 | 35.13 | 29 | 1313 |
| usa_tv | 10 | 107474 | 3561 | 3575 | 3575 | 3575 | 0 | 0.00 | 20.06 | 15.04 | 35.11 | 28 | 1181 |
| youtube_gold | 1 | 670488 | 66919 | 67021 | 67444 | 67444 | 423 | 0.00 | 20.78 | 15.15 | 35.95 | 2 | 209 |
| youtube_gold | 2 | 116740 | 66960 | 66964 | 67351 | 67351 | 387 | 0.00 | 20.21 | 15.18 | 35.42 | 1 | 204 |
| youtube_gold | 3 | 26226 | 66909 | 67001 | 67280 | 67280 | 279 | 0.00 | 20.39 | 15.14 | 35.56 | 2 | 197 |
| youtube_gold | 4 | 777573 | 66970 | 66984 | 67322 | 67322 | 338 | 0.00 | 20.21 | 15.16 | 35.39 | 1 | 204 |
| youtube_gold | 5 | 288390 | 66889 | 66984 | 67278 | 67278 | 294 | 0.00 | 20.19 | 15.09 | 35.30 | 1 | 204 |
| youtube_gold | 6 | 256788 | 66919 | 66993 | 67361 | 67361 | 368 | 0.00 | 20.25 | 15.15 | 35.41 | 2 | 197 |
| youtube_gold | 7 | 234054 | 66988 | 67001 | 67314 | 67314 | 313 | 0.00 | 20.22 | 15.14 | 35.38 | 2 | 207 |
| youtube_gold | 8 | 146317 | 66803 | 67021 | 67309 | 67309 | 288 | 0.00 | 20.20 | 15.17 | 35.40 | 2 | 209 |
| youtube_gold | 9 | 772247 | 66825 | 67000 | 67339 | 67339 | 339 | 0.00 | 20.15 | 15.13 | 35.31 | 2 | 206 |
| youtube_gold | 10 | 107474 | 66826 | 67001 | 67426 | 67426 | 425 | 0.00 | 20.26 | 15.10 | 35.38 | 2 | 190 |
| youtube_premium | 1 | 670488 | 22641 | 23297 | 47224 | 47224 | 23927 | 0.00 | 20.21 | 15.12 | 35.34 | 4 | 862 |
| youtube_premium | 2 | 116740 | 22601 | 23635 | 47433 | 47433 | 23798 | 0.00 | 20.14 | 15.11 | 35.26 | 5 | 891 |
| youtube_premium | 3 | 26226 | 22340 | 23579 | 46673 | 46673 | 23094 | 0.00 | 20.29 | 15.12 | 35.43 | 5 | 838 |
| youtube_premium | 4 | 777573 | 22352 | 23535 | 46407 | 46407 | 22872 | 0.00 | 20.29 | 15.10 | 35.41 | 5 | 835 |
| youtube_premium | 5 | 288390 | 22380 | 23150 | 46643 | 46643 | 23493 | 0.00 | 20.15 | 15.12 | 35.29 | 4 | 850 |
| youtube_premium | 6 | 256788 | 22443 | 23522 | 47093 | 47093 | 23571 | 0.00 | 20.17 | 15.13 | 35.32 | 5 | 913 |
| youtube_premium | 7 | 234054 | 22560 | 23417 | 47759 | 47759 | 24342 | 0.00 | 20.16 | 15.12 | 35.30 | 4 | 860 |
| youtube_premium | 8 | 146317 | 22602 | 23143 | 46649 | 46649 | 23506 | 0.00 | 20.13 | 15.12 | 35.26 | 4 | 858 |
| youtube_premium | 9 | 772247 | 22421 | 23462 | 46452 | 46452 | 22990 | 0.00 | 20.14 | 15.12 | 35.29 | 4 | 833 |
| youtube_premium | 10 | 107474 | 22347 | 23367 | 45943 | 45943 | 22576 | 0.00 | 20.16 | 15.17 | 35.35 | 4 | 626 |

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

Opsioni `4` ekzekuton pipeline-in final `Branch and Bound -> Genetic Algorithm -> ILS` për instancën e zgjedhur. Për çdo instance bëhen `10` runs, ndërsa për çdo run ruhen score-t dhe koha e tri fazave në `summary.json`.

Output-et ruhen në:

```text
data/output/bnb_ga_ils/<instance>
```

Për ekzekutim me parametrat finalë nga terminali mund të përdoret:

```powershell
python main.py --pipeline-runs 10 --bnb-time 30 --pipeline-ga-time 240 --ils-time 60 --pipeline-ga-profile single
```
