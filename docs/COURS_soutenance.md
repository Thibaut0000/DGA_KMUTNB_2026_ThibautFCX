# Cours — Expliquer et défendre mon projet (préparation soutenance)

> But : te rendre capable d'**expliquer** ton projet à n'importe quel niveau et de **défendre**
> tes choix et tes contributions face à un jury. Tout est en français, avec les chiffres exacts à
> connaître et les réponses aux questions pièges. Lis la Partie 8 (Q&R) au moins deux fois.
>
> Règle d'or de la soutenance : **assume ce que tu as mesuré, hedge le reste.** Un jury respecte
> beaucoup plus « voici la limite et pourquoi ce n'est pas bloquant » que des affirmations trop fortes.
>
> **Version du 16 juillet — alignée sur le papier final.** L'histoire a changé en cours de projet
> (c'est une force, pas une faiblesse) : la méthode déployée est **linéaire** (l'autoencodeur est un
> résultat négatif documenté), et la découverte du **biais de surveillance** est une contribution à
> part entière.

## Sommaire
- [1. Le pitch en 3 niveaux (30 s / 2 min / 5 min)](#p1)
- [2. Le contexte et le problème](#p2)
- [3. La question de recherche et l'objectif](#p3)
- [4. Les données](#p4)
- [5. La méthode (chaque brique, pour être défendue)](#p5)
- [6. Les contributions (le cœur de la soutenance)](#p6)
- [7. Les résultats (les chiffres par cœur)](#p7)
- [8. Questions pièges du jury + réponses](#p8)
- [9. Honnêteté scientifique (défendre les limites)](#p9)
- [10. Glossaire de défense + conseils oral](#p10)

---

<a name="p1"></a>
## 1. Le pitch en 3 niveaux

**30 secondes (une phrase) :**
> « Je classe une flotte de 628 transformateurs du plus au moins à risque, à partir des gaz dissous
> dans leur huile, **sans aucune étiquette de panne** — et en chemin j'ai découvert que la façon dont
> tout le monde **valide** ce genre de modèle est biaisée : elle mesure en partie l'*attention* des
> opérateurs, pas la santé des machines. »

**2 minutes (le résumé) :**
> Les transformateurs de puissance sont des équipements critiques et chers. On surveille leur santé
> par la DGA (analyse des gaz dissous) : un défaut naissant produit des gaz caractéristiques. Les
> méthodes classiques (Duval, IEC, Rogers) sont des **règles d'expert**, et le machine learning
> supervisé exige des **étiquettes** que les exploitants n'ont pas. Mon projet apporte trois choses,
> toutes **sans labels**. Un : représenter chaque mesure par les **log-ratios de sa composition
> gazeuse** sépare le *type* de défaut de sa *gravité* par construction — une simple **projection
> linéaire** retrouve alors les types de défaut experts (ARI **0,16 → 0,55**), et le réseau de
> neurones que j'avais construit **n'apporte rien** (résultat négatif assumé). Deux : les notes de
> terrain utilisées pour valider ces modèles sont **confondues par un biais de surveillance** —
> compter combien de fois une unité a été échantillonnée prédit les « pannes » aussi bien que toute
> ma chimie (AUC 0,76 vs 0,74, non significatif) ; je quantifie ce biais, je répare une partie du
> label en traduisant les notes thaïes, et je propose une cible d'évaluation fondée sur la chimie.
> Trois : un **indice de risque** transparent classe la flotte utilement (27 % de pannes dans le
> décile le plus risqué vs 2 % dans le plus sûr), toujours annoncé **avec** le plancher du biais.

**5 minutes (la version détaillée) :** enchaîne Partie 2 (problème) → Partie 3 (question) → Partie 5
(méthode) → Partie 6 (contributions) → Partie 7 (résultats) → une phrase de limite honnête (Partie 9).
C'est exactement le plan de ta présentation orale.

---

<a name="p2"></a>
## 2. Le contexte et le problème

Ce qu'il faut savoir poser, dans l'ordre :

1. **L'enjeu.** Un transformateur de puissance est un actif **critique et cher** (souvent > 1 M€,
   remplacement en mois). Une panne = coupure, risque incendie, pertes.
2. **L'outil de surveillance : la DGA.** Sous contrainte thermique ou électrique, l'huile et le papier
   isolant se décomposent et libèrent des **gaz dissous** (H2, CH4, C2H4, C2H2, CO/CO2). On mesure ces
   gaz en ppm : c'est la « prise de sang » du transformateur, qui détecte les défauts **naissants**.
3. **Les méthodes conventionnelles (Duval, IEC 60599, Rogers).** Des **règles fixes** sur des ratios
   ou des proportions de gaz. Limites : besoin d'un **expert**, cas **« non déterminés »**, mauvaise
   gestion des frontières et des défauts mixtes.
4. **Le vrai problème opérationnel.** Un exploitant doit **prioriser la maintenance sur toute une
   flotte** — donc **classer les unités par risque** — mais il n'a **pas d'étiquettes** de panne.
5. **Le problème caché (ma découverte).** Même pour *évaluer* un modèle sans labels, on se rabat sur
   les **notes de terrain** des opérateurs… qui enregistrent en partie leur *attention* (ils
   échantillonnent plus les unités qui les inquiètent) — un **biais de surveillance**.

**Phrase-clé du problème :** « Comment classer une flotte par risque, sans étiquettes — et comment
*valider* ce classement quand le seul label disponible est lui-même biaisé ? »

---

<a name="p3"></a>
## 3. La question de recherche et l'objectif

**Question de recherche (à savoir réciter) :**
> Peut-on classer les transformateurs d'une flotte par risque de défaut naissant et retrouver les
> **types** de défaut, **sans labels**, à partir de l'historique DGA de chaque unité — et peut-on
> faire confiance à la manière dont ces modèles sont habituellement **validés** ?

**Pourquoi c'est difficile :**
- **Pas de vérité terrain** : le supervisé est inapplicable ; l'objectif est un **classement**
  (ranking), validé indirectement.
- **Données longitudinales, déséquilibrées, multilingues** : gaz très asymétriques, arcs rares,
  notes de terrain à 69 % en thaï.
- **Le label de validation est piégé** : c'est devenu un objet d'étude en soi.

**Évolution assumée vs le brief initial :** le brief visait une représentation « temporelle
auto-supervisée » ; l'investigation a montré que le signal temporel apparent du label était en réalité
le **biais d'échantillonnage**. Le projet a pivoté — c'est documenté dans le cahier de labo — vers ce
qui était démontrable et honnête. **Si le jury demande :** « le pivot est le résultat ; poursuivre le
plan initial aurait produit un modèle validé sur un artefact. »

---

<a name="p4"></a>
## 4. Les données (savoir les décrire en 20 secondes)

- **628 transformateurs**, **4 563 mesures** d'huile (main tank), 42 fabricants, 2019–2024 (~5 ans,
  ~1 mesure / 6 mois) — un jeu **longitudinal** réel, assemblé de sources opérationnelles multiples
  en Thaïlande et **anonymisé** (un extrait anonymisé est publié avec le code).
- **391 mesures portent une note de terrain** (trip Buchholz, bushing explosé, p.f. élevé…) — **69 %
  contiennent du thaï** ; filtrées par mots-clés, elles donnent un label faible sur **71 unités**.
- **Quirks** (à mentionner pour montrer le sérieux) : gaz stockés en **texte**, très asymétriques
  (→ log1p), un outlier grossier (O2 ≈ 7×10⁸ ppm), C2H2 nul dans **97,4 %** des mesures.

---

<a name="p5"></a>
## 5. La méthode — chaque brique, expliquée pour être défendue

**Le pipeline en une phrase :** historique DGA par unité → prétraitement → **représentation
compositionnelle** (CLR + projection linéaire) pour le *type* → **indice de santé/risque**
(gravité + acétylène + temporel + anomalie) pour le *classement* → validation **avec contrôle du
biais de surveillance**.

### 5.1 Prétraitement — *quoi + pourquoi*
- **log1p** sur les gaz (asymétrie), **standardisation**, **clipping** des outliers, manquants à 0
  (≈ sous le seuil de détection).

### 5.2 La représentation compositionnelle — *le cœur technique*
- **Le piège de la sévérité (mon 1er résultat) :** tout modèle nourri de concentrations brutes apprend
  « **combien** de gaz » (la gravité), pas « **quel** défaut ». Preuve : le gaz total se lit
  linéairement dans le latent d'un AE brut (**R² = 0,63**) et l'accord avec Duval est quasi nul
  (ARI 0,14–0,16).
- **La solution :** séparer **magnitude** (gravité) et **composition** (proportions = type), et
  n'encoder que la composition via la transformation **CLR** (centred log-ratio, l'outil standard des
  données compositionnelles — Aitchison 1982). Invariance à la gravité **par construction**.
- **L'instanciation déployée est LINÉAIRE :** CLR → standardisation → **PCA 2D** → KMeans (k = 7).
  Accord avec Duval : **ARI 0,545 ± 0,002**. Déterministe, interprétable (axes = combinaisons de
  log-ratios lisibles), stable dans le temps (appris sur pré-2022, évalué sur 2022+ : ARI 0,45).
- **L'autoencodeur (SD-CAE) est un résultat négatif :** 0,47 ± 0,06, *moins bien* que la PCA, avec
  25× plus de variance ; la variante adversariale dégrade encore (0,30). Les deux sont **publiés
  comme négatifs** dans le papier.

### 5.3 L'indice de santé/risque — *le livrable opérationnel*
Quatre composantes transparentes, **sans labels** : **gravité** (H2 en tête), **acétylène** (C2H2 =
arc = le plus dangereux, pondéré ×2 par choix de domaine), **temporel** (vitesse de génération H2,
ppm/an, style C57.104), **anomalie** (erreur de reconstruction AE + Isolation Forest).

### 5.4 Le protocole d'évaluation — *ce qui rend le projet crédible*
- **Plancher de confondant co-annoncé** : l'AUC du simple compteur d'échantillons (0,76) est toujours
  affichée à côté de celle de l'indice (0,74).
- **Hold-out temporel strict** : score appris sur la 1re moitié de l'historique → prédire les
  événements de la 2de ; puis résidualisation sur le compteur.
- **Cible sans confondant** : l'**apparition d'arc** (C2H2 > 2 ppm sous 2 ans chez une unité qui n'en
  a pas) — définie par la **chimie**, décorrélée du compteur (corr −0,01).
- **Réparation du label** : traduction et classification des **211 notes thaïes distinctes**.

### 5.5 Les mots techniques à savoir définir en une phrase
- **CLR** : décrire un échantillon par ses **proportions** (log-ratios) — enlève l'effet quantité.
- **PCA** : projection linéaire qui garde le maximum de variance ; ici, 2 axes lisibles.
- **AUC** : probabilité qu'une unité à risque soit mieux classée qu'une saine (0,5 = hasard).
- **ARI** : accord entre deux découpages (mes groupes vs Duval), corrigé du hasard.
- **Biais de surveillance** : « plus on regarde, plus on trouve » — le label enregistre l'attention.
- **Bootstrap bloqué par unité** : intervalle de confiance qui respecte le fait que les points d'une
  même unité ne sont pas indépendants.

---

<a name="p6"></a>
## 6. Les contributions (LE cœur de la soutenance)

### C1 — Une représentation compositionnelle « severity-disentangled », label-free
- **Énoncé :** le CLR de la composition gazeuse sépare type et gravité par construction ; une simple
  projection linéaire retrouve la structure des types Duval **sans étiquettes**.
- **Preuve :** ARI **0,16 → 0,55** (×3,4) ; fuite de gravité supprimée ; stable en temporel (0,45) ;
  le neuronal et l'adversarial sont des **négatifs documentés**.
- **Nouveauté (hedgée) :** « à notre connaissance, la géométrie log-ratio d'Aitchison n'avait jamais
  été appliquée à la DGA ; le plus proche (Dukarm 2020) reconnaît le simplexe mais garde des
  coordonnées euclidiennes. »
- **Défense :** « ARI = accord avec une **règle** (Duval), pas une accuracy ; je revendique le gain
  relatif, la construction sans labels, et la simplicité déployable. »

### C2 — Le biais de surveillance dans la validation DGA (la contribution la plus originale)
- **Énoncé :** le label « notes de terrain » est confondu : le simple **compteur d'échantillons** fait
  aussi bien que la chimie (0,76 vs 0,74, p = 0,58) ; la validité *prospective* du score disparaît
  quand on contrôle le compteur (AUC 0,50) ; une cible **chimique** (apparition d'arc) reste
  génuinement prédictible (0,64) quand le compteur ne l'est pas (0,49).
- **Position littérature :** phénomène connu en épidémiologie/dossiers médicaux (Sackett 1979,
  Haut & Pronovost 2011, Goldstein 2016 — qui conditionne exactement sur le *nombre de visites*) ;
  **jamais rapporté pour la DGA** à notre connaissance. On **importe et quantifie** un confondant
  connu dans un domaine neuf — c'est plus défendable que « on a inventé un biais ».
- **Réparation :** +6 unités d'événements retrouvées dans les notes thaïes ; sur le label réparé,
  l'avantage du compteur **disparaît** (0,734 vs 0,735).
- **Recommandation au domaine :** tout modèle de risque DGA validé sur des événements opérateur
  devrait **co-publier le plancher du compteur**.

### C3 — L'indice de risque label-free, annoncé honnêtement
- **Preuve :** décile le plus risqué ≈ **27 %** d'événements vs ≈ 2 % pour le plus sûr ; ablation :
  gravité 0,64 → +acétylène 0,71 → +temporel 0,73 → +anomalie 0,74.
- **Défense :** « toujours lu contre le plancher 0,76 ; les composantes ont été choisies pendant le
  développement avec ce même label, donc l'AUC absolue porte un optimisme de conception — c'est
  l'ordre des composantes qui est le message, pas la 3e décimale. »

---

<a name="p7"></a>
## 7. Les résultats — les chiffres à connaître par cœur

| Résultat | Chiffre | Comment le dire honnêtement |
|----------|---------|------------------------------|
| Type de défaut (C1) | ARI **0,16 → 0,55** (PCA linéaire) | accord avec une règle, pas une accuracy ; gain ×3,4 |
| Le neuronal n'aide pas | AE 0,47 ± 0,06 < PCA 0,545 ± 0,002 | résultat **négatif publié** — la géométrie fait le travail |
| Robustesse temporelle | ARI 0,45 (appris <2022, évalué 2022+) | la structure généralise |
| Classement de risque (C3) | AUC **0,74** (IC 0,68–0,80) | **toujours** avec le plancher ci-dessous |
| Le plancher du biais (C2) | compteur d'échantillons : AUC **0,76** | différence non significative (p = 0,58) |
| Décomposition du signal | physique au-delà du compteur : p = 0,003 ; compteur au-delà de la physique : p = 9·10⁻¹³ | « dominé par l'attention, mais pas un pur artefact » |
| Validité prospective | 0,54 → **0,50** une fois le compteur contrôlé (compteur seul : 0,68) | pas de prédiction *future* démontrable avec ces features |
| Cible chimique | AUC 0,64 (IC bloqué 0,52–0,77), compteur 0,49 | **17 unités** d'apparition d'arc → intervalles larges, assumés |
| Notes thaïes | 69 % des notes ; 211 traduites ; **+6 unités** ; 0,734 vs 0,735 | une partie du biais = label incomplet |
| Déciles | ~27 % (D1) vs ~2 % (D10) | le classement est opérationnellement utile |

**À retenir absolument :** ARI **0,55 vs 0,16** (et « le neuronal fait moins bien : 0,47 »),
AUC **0,74 vs le plancher 0,76 (non significatif)**, **17 unités** pour la cible chimique, et
« je n'ai pas de vérité terrain : je valide indirectement et je le dis ».

---

<a name="p8"></a>
## 8. Questions pièges du jury + réponses (à répéter à voix haute)

**Q. Pourquoi ne pas faire du supervisé, plus précis ?**
R. Pas d'étiquettes fiables à l'échelle de la flotte — et ma deuxième contribution montre que même le
label *faible* disponible est biaisé. Le supervisé sur ces notes apprendrait l'attention des
opérateurs.

**Q. Pourquoi un autoencodeur plutôt qu'une simple PCA ?** *(question inversée — c'est un piège pour
qui n'a pas lu le papier)*
R. **Nous avons testé exactement ça — et la PCA gagne** (0,545 vs 0,474, avec 25× moins de variance).
C'est pourquoi la méthode déployée est linéaire et que l'autoencodeur est publié comme **résultat
négatif**. La leçon : une fois la bonne géométrie choisie (CLR), la complexité du modèle n'apporte
rien sur 4 500 échantillons en 5 dimensions.

**Q. Votre AUC 0,74 est SOUS le compteur 0,76 — votre modèle ne sert donc à rien ?**
R. Trois réponses. (1) La différence est **non significative** (p = 0,58) : égalité statistique.
(2) Un test log-vraisemblance montre que la physique apporte un signal **réel au-delà du compteur**
(p = 0,003). (3) Surtout : c'est **exactement le message du papier** — ce label ne permet pas de
départager, d'où la cible chimique sans confondant, où la chimie prédit (0,64) et le compteur non
(0,49). Le « problème » de mon indice est ma deuxième contribution.

**Q. Un ARI de 0,55, c'est faible, non ?**
R. C'est un accord avec une **règle imparfaite** (Duval), pas une accuracy — et Duval Triangle 1
sur-étiquette PD sur cette flotte. Le message est le **gain ×3,4** sans étiquettes et la stabilité
temporelle (0,45).

**Q. Seulement 17 unités pour votre cible chimique — c'est sérieux ?**
R. C'est précisément pourquoi je rapporte des **intervalles bloqués par unité** ([0,52 ; 0,77] —
au-dessus du hasard, de justesse) au lieu du test naïf (anti-conservateur, que j'avais d'abord utilisé
et que j'ai **corrigé moi-même**). Résultat robuste sur 9 variantes seuil × horizon (0,64–0,73),
annoncé comme modeste.

**Q. Vous avez « découvert » le biais de surveillance ? Il est connu depuis les années 70…**
R. Exact — Sackett 1979, Haut & Pronovost 2011, Goldstein 2016 en épidémiologie, tous cités. Ma
contribution est de l'**identifier, le quantifier et le contrôler pour la DGA**, où — à notre
connaissance — il n'avait jamais été rapporté alors que tout le domaine valide sur ces labels.

**Q. Pourquoi les proportions (CLR) plutôt que les concentrations brutes ?**
R. Le **type** de défaut est une propriété **compositionnelle** (le triangle de Duval travaille en %).
Les concentrations portent surtout la gravité. Le CLR est l'outil mathématiquement correct pour des
proportions (géométrie d'Aitchison) ; preuve empirique : ARI ×3,4.

**Q. Pourquoi l'acétylène a un poids spécial ?**
R. Le C2H2 ne se forme qu'à très haute énergie → il signale l'**arc**, le défaut le plus dangereux,
même en faible quantité. Pondération ×2 **par choix de domaine** (pas ajustée sur le label ; la
sensibilité au poids est rapportée).

**Q. Le brief parlait de représentation « temporelle » — où est-elle ?**
R. C'est le pivot assumé : le signal « temporel » le plus fort du label était le **biais
d'échantillonnage** lui-même. Le temporel honnête restant : la vitesse de génération H2 dans l'indice
(+0,02 AUC), la stabilité temporelle de la carte (0,45), et le hold-out prospectif — négatif et
publié. Des features de trajectoire dans le plan CLR-PCA sont la suite naturelle.

**Q. Qu'est-ce qui empêche la généralisation à d'autres flottes ?**
R. Rien ne la garantit : une flotte, un opérateur. C'est écrit dans le papier (« in this fleet »).
La méthode de contrôle, elle, est transférable telle quelle — c'est la recommandation au domaine.

**Q. Vos notes thaïes : traduites par qui, validées comment ?**
R. Traduites et classifiées note par note (211 notes distinctes), table publiée avec gloses anglaises
et règles d'exclusion (les suivis « C2H2 trouvé » sont exclus car circulaires) ; **en attente de
validation domaine finale** — dit tel quel dans le papier.

---

<a name="p9"></a>
## 9. Honnêteté scientifique — défendre les limites (sans s'excuser)

1. **Une seule flotte** → « la prévalence du biais ailleurs est une question ouverte et testable ;
   la méthode de contrôle est portable. »
2. **Labels faibles** → « c'est devenu un objet d'étude ; j'ai réparé ce qui était réparable
   (thaï) et construit une cible sans confondant. »
3. **ARI modéré, δ-sensible** → « sensibilité au remplacement des zéros divulguée (0,46–0,55 pour
   δ ≤ 10⁻³) ; C2H2 est nul à 97 %, c'est structurel. »
4. **Optimisme de conception de l'indice** → « composantes choisies avec le label en boucle ;
   dit explicitement dans le papier ; l'ordre des composantes est le message. »
5. **Négatifs publiés** → SD-CAE, adversarial, Deep SVDD (s'effondre), VAE/contrastif, RUL/Cox
   (C-index 0,565, n.s.) : « chaque négatif épargne du temps au suivant. »

**La phrase qui clôt bien :** « Ce projet livre un classement utilisable et, surtout, une méthode pour
ne pas se mentir en le validant. »

---

<a name="p10"></a>
## 10. Glossaire de défense + conseils oral

- **DGA** : analyse des gaz dissous — la prise de sang du transformateur.
- **Duval / IEC / Rogers** : règles d'expert sur ratios/proportions de gaz.
- **CLR / Aitchison** : log-ratios centrés — la géométrie correcte des proportions.
- **ARI / AUC / IC / p** : voir §5.5 ; toujours donner l'intervalle avec le chiffre.
- **Confound floor** : l'AUC du compteur d'échantillons, à co-publier avec toute AUC sur notes.
- **Bootstrap bloqué** : IC respectant la non-indépendance intra-unité.

**Conseils oral :**
- Montre en priorité : la **carte CLR-PCA** (types sans labels), le **panneau 3 volets** du biais,
  et les **déciles** de risque. Ce sont tes trois images.
- Connais **4 chiffres** par cœur : **0,55 vs 0,16** · **0,74 vs 0,76 (p = 0,58)** · **17 unités,
  IC [0,52 ; 0,77]** · **27 % vs 2 %**.
- Si une question te coince : « bonne question — voici ce que j'ai mesuré, voici ce que je ne peux
  pas conclure, et voici comment je le testerais. » C'est une réponse de chercheur.
- La démo : lance le dashboard **avant** la soutenance (localhost:8501) ; si elle échoue, chaque vue
  existe en figure dans le papier.
