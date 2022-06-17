# :heart::computer:Grafovske Neuronske Mreže:computer::heart:

Projekat ima za cilj da predstavi Grafovske neuronske mreže i pokaže primenu i kvalitet različitih modela korišćenih za klasifikaciju podataka na poznatom skupu **Cora**. U tu svrhu implementirana su dva naučna rada na ovu temu - [GAT](https://arxiv.org/abs/1710.10903) i [GCN](https://arxiv.org/abs/1609.02907).

Projekat je rađen u sklopu kursa Mašinsko učenje na Matematičkom fakultetu, Univerziteta u Beogradu.

## Sadržaj

### 1. Teorija

U ovom poglavlju uvodimo osnovne elemente teorije grafova, koji čine osnovu na kojoj se zasniva oblast grafovskih neuronskih mreža. Objašnjeni su koncepti konvolutivnih grafovskih neuronskih mreža kao i grafovskih mreža sa mehanizmom pažnje.

### 2. Skup podataka i vizualizacija

Predstavljen je i analiziran skup podataka Cora sa kojim smo radili u okviru projekta. Prikazane su osnovne statistike poput broja čvorova i grana kao i izvedene poput broja suseda.

### 3. Trening i analiza

Istrenirana su tri različita modela - višeslojni perceptron, konvolutivna grafovska neuronska mreža i grafovska mreža sa mehanizmom pažnje. Modeli su upoređeni i izvršena je topološka analiza rezultata gde je prikazano šta su modeli zapravo naučili.

## Pokretanje
Projekat je implementiran u sklopu jupyter svesaka, pa je preporučeno sprovesti sledeće korake u korenu repozitorijuma:

1. `$ conda env create --file environment.yml`
2. `$ conda activate mlmatf-gnn`
3. `$ jupyter notebook`