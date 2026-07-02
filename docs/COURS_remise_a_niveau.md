# Cours de remise à niveau — tout ce qu'il faut pour le projet

> Objectif : te donner, en un seul document, toutes les notions pour comprendre et mener
> ton projet (détection non-supervisée de défauts de transformateurs par DGA). Tu es à
> l'aise en Python/ML, donc on va vite sur le code et on prend le temps sur **le domaine**
> (transformateurs, gaz) et sur **les briques deep learning** précises du sujet.
>
> Temps de lecture : ~45 min. Avec les vidéos : ~2 h. Lis dans l'ordre une première fois ;
> reviens ensuite par section. Les questions de fin sont dans le chat.

## Sommaire
- [Partie 0 — La vue d'ensemble](#partie-0)
- [Partie 1 — Le domaine : transformateurs & DGA](#partie-1)
- [Partie 2 — Fondations ML pour ce projet](#partie-2)
- [Partie 3 — Deep learning : autoencoders & VAE](#partie-3)
- [Partie 4 — Le non-supervisé en aval : clustering, anomalies, métriques](#partie-4)
- [Partie 5 — Comment tout s'assemble dans TON projet](#partie-5)
- [Partie 6 — Glossaire express](#partie-6)
- [Toutes les vidéos](#videos)

---

<a name="partie-0"></a>
## Partie 0 — La vue d'ensemble (à garder en tête)

Tout le projet tient en une chaîne. Garde cette image : chaque notion du cours est une étape ici.

```
Mesures DGA brutes        Prétraitement            Autoencoder              En aval
(H2, CH4, C2H2, ...)  →   clip → log → scale   →   encodeur → LATENT   →   ┌── Clustering  → "patterns de défauts"
   ~4500 échantillons         (Partie 2)            décodeur (Partie 3)    └── Erreur de reconstruction → "anomalies"
                                                                                      │
                                                                          Évaluation (Partie 4) :
                                                                          interne (silhouette) + externe (vs Duval)
```

**L'idée centrale du sujet :** au lieu d'appliquer des règles d'expert (Duval/IEC), on laisse
un réseau de neurones **apprendre tout seul** une représentation compacte des gaz, puis on
**regroupe** (clustering) pour découvrir des familles de défauts et on **repère les anomalies**
par l'erreur de reconstruction — **sans étiquettes**. On valide ensuite en comparant aux
méthodes conventionnelles.

> **La question de recherche que tu vas creuser** (déjà visible sur le 1ᵉʳ test) : ton espace
> latent sépare-t-il les défauts par **type** (PD, thermique, arc…) ou seulement par **gravité
> globale** du dégazage ? Le premier résultat penche vers « gravité » → piste : normaliser par
> échantillon (proportions de gaz, comme Duval). Garde ça en tête tout le long.

---

<a name="partie-1"></a>
## Partie 1 — Le domaine : transformateurs & DGA

C'est la partie la plus neuve pour un profil INFO. Prends ton temps ici.

### 1.1 Le transformateur de puissance, en 2 minutes
Un transformateur élève ou abaisse la tension du réseau électrique (ex. 115 kV ↔ 22 kV dans
tes données). C'est un actif **critique et cher** (souvent > 1 M€, des mois de délai de
remplacement) : une panne = coupure + risque incendie. À l'intérieur d'une grande cuve
(« main tank », d'où le nom de ton fichier) :
- des **enroulements** de cuivre autour d'un **noyau** de fer ;
- le tout baigne dans une **huile minérale isolante** qui sert à la fois d'**isolant
  électrique** et de **refroidissement** ;
- les enroulements sont isolés par du **papier/carton (cellulose)**.

Retiens deux matériaux qui se dégradent et produisent des gaz : **l'huile** et **le papier**.

### 1.2 Pourquoi un transformateur « produit des gaz »
En service, le transfo subit deux familles de contraintes :
- **thermiques** (points chauds, surcharge, mauvais contact) ;
- **électriques** (décharges partielles, étincelles, **arcs**).

Sous ces contraintes, les molécules d'huile et de papier se **cassent** (décomposition).
Cette casse libère des **gaz dissous** dans l'huile. Le point clé : **le type et la quantité
de gaz dépendent du type et de la sévérité du défaut.** Un échauffement modéré et un arc
violent ne produisent pas les mêmes gaz.

### 1.3 La DGA : la « prise de sang » du transformateur
**Dissolved Gas Analysis** = on prélève un échantillon d'huile, on extrait les gaz dissous et
on les mesure (chromatographie en phase gazeuse). On obtient des concentrations en **ppm**
(parties par million) — exactement les colonnes de ton fichier. L'intérêt : détecter des
**défauts naissants** (« incipient ») **avant** la panne. C'est la méthode de référence pour
surveiller un transfo en service.

Video: *Domaine (à voir en priorité, c'est ton point faible) :*
- [Ultimate Step-by-Step Guide to DGA for Transformers](https://www.youtube.com/watch?v=BewdZ4yPlNY)
- [DGA + Triangle de Duval (Part 1/2)](https://www.youtube.com/watch?v=_-ByAf7HHBw)
- Texte clair : [Megger — How to interpret DGA results](https://www.megger.com/en/knowledge-hub/how-to-interpret-dga-results-for-transformer-health)

### 1.4 Les gaz et ce qu'ils racontent (relie ça à tes colonnes)
Règle physique simple : **plus le défaut est énergétique/chaud, plus les liaisons se cassent**,
et plus on va vers des molécules « à triple liaison » comme l'acétylène.

| Gaz | Nom | Signification dominante |
|-----|-----|--------------------------|
| **H2** | hydrogène | **décharges partielles (PD)** ; premier gaz produit, le plus « facile » |
| **CH4** | méthane | défaut **thermique basse température** |
| **C2H6** | éthane | défaut **thermique basse température** |
| **C2H4** | éthylène | défaut **thermique haute température** (> 700 °C) |
| **C2H2** | acétylène | **arc / décharge haute énergie** — le gaz « grave », n'apparaît qu'à très haute T° |
| **CO, CO2** | mon/dioxyde de carbone | dégradation du **papier (cellulose)** |
| O2, N2 | oxygène, azote | **atmosphériques** (viennent de l'air) → non diagnostiques, **exclus** |
| TCG | total combustible gas | **somme** d'autres gaz → exclu (fuite d'information / « leakage ») |

C'est pour ça que `config/default.yaml` garde `H2, CH4, C2H2, C2H4, C2H6, CO, CO2` comme
*features* et exclut O2/N2/TCG.

### 1.5 Les méthodes conventionnelles (= tes *baselines*)
Avant le ML, les experts utilisent des **règles** :
- **Key Gas** : on regarde le gaz dominant.
- **Méthodes par ratios** (Rogers, **IEC 60599**, Doernenburg) : on calcule des rapports entre
  gaz (ex. C2H2/C2H4, CH4/H2, C2H4/C2H6) et une table de seuils donne le type de défaut.
- **Triangle de Duval** : on prend les **proportions** de CH4, C2H4 et C2H2 (elles somment à
  100 %) et on place le point dans un triangle découpé en **zones** = types de défauts.

Ces méthodes sont codées dans `src/dga/conventional.py` — tu les exécutes telles quelles comme
référence. Video: [Le Triangle de Duval en 3 min (Reinhausen)](https://www.reinhausen.com/the-duval-triangle-explained-in-3-minutes)

### 1.6 Les 7 classes de défauts (le vocabulaire cible)
| Code | Défaut |
|------|--------|
| **PD** | décharge partielle (partial discharge) |
| **D1** | décharge de **basse** énergie (étincelles) |
| **D2** | décharge de **haute** énergie (**arc**) |
| **T1** | défaut **thermique < 300 °C** |
| **T2** | défaut thermique **300–700 °C** |
| **T3** | défaut thermique **> 700 °C** |
| **DT** | mélange thermique + électrique |

Sur tes données, Duval donne surtout PD (~1500), T1 (~1450), T2 (~690), T3 (~490) et peu de
D1/D2/DT — un **fort déséquilibre de classes**, à garder en tête pour l'évaluation.

### 1.7 Pourquoi passer au ML non-supervisé (la motivation de TON sujet)
Limites du conventionnel : règles **fixes**, besoin d'un **expert**, cas **« non déterminés »**,
mauvaise gestion des **zones frontières** et des **défauts multiples**. Et surtout : dans la
vraie vie, **on n'a pas d'étiquettes** fiables sur des milliers d'échantillons. D'où l'approche
**non-supervisée** : laisser les données révéler les structures, et signaler ce qui est anormal.

---

<a name="partie-2"></a>
## Partie 2 — Fondations ML pour ce projet

### 2.1 Supervisé vs non-supervisé
- **Supervisé** : on apprend une fonction X → y à partir d'exemples **étiquetés** (ex. classer
  une image). Besoin de `y`.
- **Non-supervisé** : pas de `y`. On cherche la **structure** des données : groupes (clustering),
  représentation compacte (réduction de dimension), points anormaux (anomalies).
- **Ton cas** : pas de `y` fiable → non-supervisé. Astuce du projet : les diagnostics Duval/IEC
  servent de **labels faibles** pour **évaluer a posteriori** (pas pour entraîner), et les
  **notes de terrain** (391 lignes : « Buchholz trip », bushing explosé…) valident les anomalies.

### 2.2 Tes données et le prétraitement — le « pourquoi » de chaque étape
Tes features sont des concentrations en ppm. Trois propriétés gênantes (vues à l'EDA) :
1. **Très asymétriques** (right-skew) : médianes minuscules, maxima énormes. Beaucoup de zéros.
2. **Outliers de saisie** : un O2 à 7×10⁸ ppm (impossible) → erreur de saisie.
3. Échelles **très différentes** d'un gaz à l'autre.

Le pipeline (`src/dga/preprocessing.py`) répond à chaque point :
- **Clipping** au quantile 0.999 → neutralise les valeurs absurdes.
- **Imputation à 0** des manquants → ≈ « sous le seuil de détection ».
- **`log1p`** (= log(1+x)) → **compresse** les grandes valeurs, **étale** les petites,
  **stabilise la variance**. Indispensable sur la DGA. (On fait `1+x` pour gérer les zéros.)
- **Standardisation** (moyenne 0, écart-type 1) → met tous les gaz **à la même échelle**, sinon
  le réseau et le clustering seraient dominés par les gaz aux grandes valeurs.

> Intuition log : sans log, passer de 1 à 10 ppm et de 1000 à 1010 ppm « pèsent » pareil en
> valeur absolue mais pas en signification. Le log rend les **rapports** comparables.

### 2.3 Représentation & espace latent (le cœur de l'idée)
**Réduire la dimension** = résumer chaque échantillon (7 gaz) par **quelques nombres** qui
gardent l'essentiel.
- **PCA** : réduction **linéaire** (combinaisons linéaires des features).
- **Autoencoder** : réduction **non-linéaire**, apprise par un réseau (Partie 3) → capture des
  relations plus riches entre gaz.

Ce résumé compact, c'est l'**espace latent** (ou « code »). On y travaille parce qu'il est plus
**propre** (débruité), plus **petit** (clustering plus facile) et **visualisable** en 2D.

---

<a name="partie-3"></a>
## Partie 3 — Deep learning : autoencoders & VAE

### 3.1 Rappel express réseau de neurones
Un **neurone** = combinaison linéaire des entrées (poids) + une **activation** non-linéaire
(ReLU…). On empile des **couches**. On définit une **fonction de perte** (l'erreur) et on
ajuste les poids par **descente de gradient** via **rétropropagation**. C'est tout ce qu'il faut
ici. Video: [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)

### 3.2 L'autoencoder (AE) — la brique n°1 du projet
Architecture en sablier :
```
entrée (7 gaz) → [ENCODEUR] → code latent (ex. 3 nombres) → [DÉCODEUR] → reconstruction (7 gaz)
```
- On entraîne le réseau à **reproduire son entrée** en sortie.
- La perte = **erreur de reconstruction** (MSE entre entrée et sortie).
- Le **goulot** (latent, petit) **force** le réseau à ne garder que l'information utile → il
  apprend une **représentation compacte** des gaz.

Deux usages, tous deux exploités dans ton projet :
1. **Le code latent** → on le donne au **clustering** (Partie 4).
2. **L'erreur de reconstruction** → un échantillon **anormal** (jamais vu) est **mal
   reconstruit** → grosse erreur → **anomalie**. (Implémenté dans `src/dga/anomaly.py`.)

Video: [Autoencoder Explained](https://www.youtube.com/watch?v=q222maQaPYo) · code : `src/dga/models/autoencoder.py`

### 3.3 Le VAE (Variational Autoencoder) — la brique n°2
Problème de l'AE simple : son espace latent peut être « troué » et irrégulier. Le **VAE** le
rend **lisse et structuré** :
- l'encodeur ne sort plus un point, mais une **distribution** (une **moyenne μ** et une
  **variance σ²**) pour chaque échantillon ;
- on **tire au hasard** dans cette distribution (le *reparameterization trick* permet de garder
  l'entraînement par gradient) ;
- la perte s'appelle l'**ELBO** = **reconstruction** + **KL** :
  - le terme de **reconstruction** : bien reproduire l'entrée (comme l'AE) ;
  - le terme **KL** (divergence de Kullback-Leibler) : **régularise** le latent en le forçant à
    ressembler à une **gaussienne standard** N(0, I) → latent compact et continu.
- **β-VAE** : on multiplie le KL par **β**. β grand → latent plus structuré/« démêlé » mais
  reconstruction moins fidèle ; β petit → l'inverse. C'est un **bouton de réglage** que tu
  testeras.

Video: [Variational Autoencoders EXPLAINED](https://www.youtube.com/watch?v=OiJM7UQw3cs) · code : `src/dga/models/vae.py`

### 3.4 AE vs VAE — lequel, quand ?
- **AE** : plus simple, très bon pour la **reconstruction** et donc la **détection d'anomalies**.
- **VAE** : latent plus **lisse**, génératif, parfois meilleur pour le **clustering**.
- Dans ton projet : tu **compares les deux** (c'est une expérience prévue en semaine 2).

---

<a name="partie-4"></a>
## Partie 4 — Le non-supervisé en aval

### 4.1 Clustering : regrouper sans étiquettes
On regroupe les codes latents en familles. Trois algos (dans `src/dga/clustering.py`) :
- **K-means** : tu fixes **k** (nb de clusters) ; trouve des clusters **sphériques** ; rapide.
  Video: [StatQuest — K-means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- **GMM** (mélange de gaussiennes) : clusters **ellipsoïdaux**, affectation **probabiliste**.
- **DBSCAN / HDBSCAN** : clusters par **densité**, **pas besoin de fixer k**, gère le **bruit**
  (points isolés). Video: [StatQuest — DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok)

### 4.2 Combien de clusters ? — métriques INTERNES
Sans étiquettes, on juge la **qualité géométrique** des clusters :
- **Silhouette** (−1 → 1, **haut = mieux**) : un point est-il bien plus proche de son cluster
  que du voisin ?
- **Davies-Bouldin** (**bas = mieux**) : clusters compacts et bien séparés ?
- **Calinski-Harabasz** (**haut = mieux**) : variance inter/intra-clusters.
- **Méthode du coude** : on trace la métrique selon k et on cherche le « coude ».

> Caution: Piège (et c'est ton 1ᵉʳ résultat) : une silhouette **très haute** avec **k=2** peut juste
> signifier que le latent code la **gravité** (gaz faibles vs forts), pas le **type** de défaut.
> Une bonne métrique interne ≠ un clustering **utile** physiquement. D'où les métriques externes.

### 4.3 Détection d'anomalies
Repérer les échantillons « anormaux » (défauts naissants) :
- **Erreur de reconstruction** (de l'AE/VAE) : l'auto-encodeur reconstruit mal ce qui sort de
  l'ordinaire → score élevé = anomalie. On **seuille** (ex. top 5 % par quantile).
- **Isolation Forest** : isole les points « seuls » avec des arbres aléatoires (peu de coupes
  pour les isoler = anormaux).
- **One-Class SVM** : apprend la frontière du « normal ».
- **Validation** : on compare les anomalies signalées aux **notes de terrain** (`has_note`).
  Comme ces notes sont **rares et non exhaustives**, on lit précision/rappel comme **indicatifs**.

Video: [Anomaly Detection : clustering, densité, Isolation Forest](https://www.youtube.com/watch?v=QZNEJHbophM)
· [Isolation Forests — identifier les outliers](https://www.youtube.com/watch?v=Y1x51i1936M)

### 4.4 Comparer aux labels Duval — métriques EXTERNES
Pour montrer que les structures découvertes sont **physiquement sensées**, on mesure l'**accord**
entre tes clusters et les diagnostics Duval/IEC (`src/dga/evaluation.py`) :
- **ARI** (Adjusted Rand Index, −1 → 1) : accord corrigé du hasard.
- **NMI** (Normalized Mutual Information, 0 → 1) : information partagée.
- **Homogénéité / Complétude** : un cluster = un seul type ? / un type = un seul cluster ?

> ARI **bas** + silhouette **haute** = clusters nets **mais** qui ne correspondent pas aux types
> Duval → exactement ton fil de recherche.

---

<a name="partie-5"></a>
## Partie 5 — Comment tout s'assemble dans TON projet

| Étape | Notion | Fichier |
|-------|--------|---------|
| Charger + nettoyer le xlsx | Partie 1 (gaz, notes) | `src/dga/data.py` |
| Clip → log1p → scale | Partie 2.2 | `src/dga/preprocessing.py` |
| Baselines Duval/IEC/Rogers | Partie 1.5–1.6 | `src/dga/conventional.py` |
| Apprendre le latent | Partie 3 (AE/VAE) | `src/dga/models/` |
| Découvrir les patterns | Partie 4.1–4.2 | `src/dga/clustering.py` |
| Repérer les anomalies | Partie 4.3 | `src/dga/anomaly.py` |
| Évaluer (interne + externe) | Partie 4.2 & 4.4 | `src/dga/evaluation.py` |

**Le fil rouge scientifique :** type vs gravité. Si le latent encode la gravité, tu testes la
**normalisation par échantillon** (proportions de gaz) pour récupérer la structure par type —
et même si ça « échoue », c'est un **résultat publiable** (tu expliques *pourquoi*).

---

<a name="partie-6"></a>
## Partie 6 — Glossaire express
- **DGA** : analyse des gaz dissous dans l'huile du transformateur.
- **ppm** : parties par million (unité des concentrations de gaz).
- **Incipient fault** : défaut naissant, détecté avant la panne.
- **Espace latent / code** : représentation compacte apprise par l'encodeur.
- **Reconstruction / MSE** : erreur entre entrée et sortie de l'autoencoder.
- **ELBO / KL** : objectif du VAE (reconstruction + régularisation du latent).
- **β-VAE** : VAE avec un poids β réglable sur le terme KL.
- **Silhouette / DBI / CH** : métriques internes de qualité de clustering.
- **ARI / NMI** : métriques externes (accord avec des labels de référence).
- **Labels faibles** : étiquettes imparfaites (ici : Duval + notes terrain) pour évaluer.
- **PD / D1 / D2 / T1 / T2 / T3 / DT** : les 7 classes de défauts DGA.

---

<a name="videos"></a>
## Toutes les vidéos (récap)

**Domaine — DGA & Duval**
- [Ultimate Step-by-Step Guide to DGA](https://www.youtube.com/watch?v=BewdZ4yPlNY)
- [DGA + Triangle de Duval (Part 1/2)](https://www.youtube.com/watch?v=_-ByAf7HHBw)
- [Le Triangle de Duval en 3 min (Reinhausen, texte)](https://www.reinhausen.com/the-duval-triangle-explained-in-3-minutes)

**Réseaux de neurones**
- [3Blue1Brown — But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk)

**Autoencoders & VAE**
- [Autoencoder Explained](https://www.youtube.com/watch?v=q222maQaPYo)
- [Variational Autoencoders EXPLAINED](https://www.youtube.com/watch?v=OiJM7UQw3cs)

**Clustering**
- [StatQuest — K-means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [StatQuest — DBSCAN](https://www.youtube.com/watch?v=RDZUdRSDOok)

**Détection d'anomalies**
- [Unsupervised Anomaly Detection (clustering, densité, Isolation Forest)](https://www.youtube.com/watch?v=QZNEJHbophM)
- [Isolation Forests — identifier les outliers](https://www.youtube.com/watch?v=Y1x51i1936M)

> Vérifie que les vidéos correspondent bien à ton besoin avant d'y passer du temps ; si une est
> indisponible, cherche le titre sur YouTube, ce sont des classiques faciles à retrouver.
