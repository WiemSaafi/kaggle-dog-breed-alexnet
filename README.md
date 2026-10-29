# 🐕 Dog Breed Identification avec AlexNet

Classification de 120 races de chiens (Kaggle [Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification)) en utilisant **uniquement AlexNet** (PyTorch), en 3 heures sur un GPU T4.

**Score Kaggle final : log loss 1,180** (soumission tardive, sans rang officiel).

## Résultats

| Soumission | Contenu | Log loss Kaggle |
|---|---|---|
| 1 | AlexNet gelé, tête `fc7` seule | 1,545 |
| 2 | Ensemble de 6 variantes d'AlexNet (80 % du train) | 1,230 |
| 3 | Même ensemble, réentraîné sur 100 % du train | **1,180** |

- Amélioration de **−24 %** entre la soumission 1 et la soumission 3
- Environ **4 fois mieux que le hasard** (ln 120 ≈ 4,79)
- Accuracy en validation du modèle final : 65,3 %
- Les scores Kaggle sont très proches de la validation (1,230 contre 1,233), donc le protocole d'évaluation est fiable

## Méthodes

- **Transfer learning** à partir des poids ImageNet
- **Résolution d'entrée augmentée** (352 px) pour compenser le filtre 11×11 de pas 4
- **TTA 10 crops** (méthode d'évaluation du papier original d'AlexNet)
- **Entraînement multi-vues** (12 vues par image) contre la mémorisation
- **Fine-tuning régularisé** (mixup, recadrages forts)
- **Ensemble pondéré** de 6 variantes d'AlexNet, l'amélioration la plus efficace (−0,12)

## Ce que j'ai appris

Le vrai problème n'était pas la taille du modèle mais la **mémorisation** : le fine-tuning naïf n'apportait rien, alors que le multi-vues, le mixup et l'ensemble ont fait baisser la log loss. Les races très proches (husky / malamute, caniches) restent les plus confondues.

## Contenu du dépôt

- `Dog_Breed_Arena_AlexNet_final_ll.ipynb` : notebook complet (exploration, benchmark des stratégies, optimisation, analyse d'erreurs, soumissions)

## Exécuter le notebook

1. Ouvrir le notebook dans Google Colab avec un GPU (T4)
2. Accepter le règlement de la compétition sur Kaggle
3. Se connecter avec un token Kaggle quand le notebook le demande
4. Exécuter les cellules de haut en bas

## Technologies

Python, PyTorch, torchvision, scikit-learn, NumPy, pandas, Matplotlib
