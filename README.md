# GTSDB Traffic Signs (Detection, Classification, Segmentation)

Projet academiqueautour du dataset GTSDB : preparation des donnees, MLP, CNN, transfert EfficientNet et autoencoder de segmentation PyTorch.
## Sommaire
- Vision rapide
- Arborescence
- Installation
- Donnees (ou les mettre)
- Pipeline complet (parties 1 -> 5)
- Resultats attendus (emplacements pour graphiques)
- Problemes frequents
- Licence

## Vision rapide
- Dataset : GTSDB (TrainIJCNN2013.zip). Images et annotations gt.txt.
- Pretraitement : extraction d'imagettes 64x64, filtrage des classes rares, splits train/val/test en JSON.
- Modeles :
  - MLP (partie 2) sur images aplaties 64x64x3.
  - CNN custom (partie 3) avec data augmentation.
  - EfficientNet finetune (partie 4, notebook).
  - Autoencoder U-Net-like pour segmentation de formes (partie 5, PyTorch).
- Scripts principaux : partie_1/main.py, partie_2/main.py, partie_3/main.py, notebooks en parties 4 et 5.

## Arborescence
- partie_1 : preparation, resize, splits, JSON.
- partie_2 : MLP Keras (build/entrainement/evaluation + courbes).
- partie_3 : CNN Keras (augmentation, confusion matrix, comparaison MLP).
- partie_4 : EfficientNet (notebook, poids efficientnet_epoch12.pt).
- partie_5 : PyTorch autoencoder (segmentation), datasets synthetiques.

## Installation (Windows, PowerShell)
1) Creer un venv
`
python -m venv venv
`
2) Activer
`
venv\Scripts\Activate.ps1
`
3) Installer les dependances
`
pip install -r requirements.txt
`

## Donnees (GTSDB)
1) Telecharger TrainIJCNN2013.zip : https://erda.ku.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html
2) Creer la hierarchie :
`
data/
  Image/          # 00000.ppm ... 00599.ppm
  gt.txt          # annotations officielles
  64x64/          # sera cree par la partie 1 (imagettes + JSON)
`
3) Verifier les chemins dans les scripts si vous changez l'arborescence.

## Pipeline complet
### Partie 1  Preparation (resize, splits)
- Executer :
`
python partie_1/main.py
`
- Effets :
  - Genere data/64x64/ avec imagettes redimensionnees.
  - Cree les JSON train.json, val.json, test.json contenant couples (path, classe) filtres (classes <20 occurrences supprimees).

### Partie 2  MLP (classification baseline)
- Lancer l'entrainement :
`
python partie_2/main.py
`
- Composants : construction du modele MLP, data augmentation basique, sauvegarde du modele et de l'historique (partie_2/models/).
- Evaluation : matrice de confusion, analyse des confusions, fonctions utilitaires dans Evaluation.

### Partie 3  CNN (amelioration des performances)
- Entrainement complet : decommentez la ligne correspondante dans partie_3/main.py puis :
`
python partie_3/main.py
`
- Par defaut, le script execute test_cnn_complet() qui charge le modele entraine, evalue et trace les metriques.
- Architecture : 3 blocs Conv/ReLU (+ MaxPool sur les deux premiers), Dense 128 + Dropout 0.5, sortie softmax.

### Partie 4  EfficientNet (transfert learning)
- Notebook : partie_4/partie_4.ipynb.
- Poids entraines : partie_4/efficientnet_epoch12.pt.
- Ouvrir dans Jupyter/VS Code pour rejouer le finetune et exporter les courbes.

### Partie 5  Segmentation (autoencoder PyTorch)
- Modele : U-Net light (encodeur Conv/Pool, decodeur Upsample + Conv) dans partie_5/model.py.
- Dataset : ShapesDataset charge images et masques (niveaux de gris) dans partie_5/dataset.py.
- Generation de donnees synthetiques : scripts dataset_generator.py et dataset_generator_multi.py.
- Notebook : partie_5/partie_5.ipynb pour entrainement, sauvegardes dans partie_5/checkpoints/.

## Resultats attendus (placeholders pour vos graphes)
- Courbes MLP : ajoutez vos figures ici (ex. docs/img/mlp_loss.png).
- Courbes CNN : idem (loss/accuracy train vs val, confusion matrix).
- EfficientNet : top-1 accuracy / F1, courbes de finetune.
- Segmentation : exemples entree -> masque predit (avant/apres entrainement).

## Problemes frequents
- ModuleNotFoundError : verifier que le venv est active et que vous lancez depuis la racine du projet.
- Chemins relatifs : adaptez image_dir, annotation_file, imagettes_dir_64x64 si vos donnees sont ailleurs.
- GPU non detecte (PyTorch) : forcer device="cpu" dans le notebook partie 5 ou installer CUDA adaptee.
- Memoire : reduire batch_size (CNN/MLP) et desactiver l'augmentation si necessaire.

## Licence
Projet pedagogique. Adapter selon vos besoins avant publication publique.

