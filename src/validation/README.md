# Thermal Validation Tool

## Vue d'ensemble

Outil de validation complet pour l'analyse thermique par éléments finis de fusées en rentrée atmosphérique. Détecte les valeurs irréalistes, identifie la physique manquante, et fournit des recommandations d'ingénierie.

## Installation

Les dépendances sont déjà installées avec le package principal:

```bash
pip install -e .
```

## Utilisation

### Validation basique

```bash
python scripts/validate_results.py
```

Valide le fichier CSV par défaut `data/output/csv/results_parametric_study.csv` avec le matériau Carbon-Carbon.

### Options avancées

```bash
# Spécifier un fichier CSV et un matériau
python scripts/validate_results.py --csv path/to/results.csv --material PICA-X

# Exporter les résultats avec drapeaux de validation
python scripts/validate_results.py --export data/output/validation/flagged_results.csv

# Lister les matériaux disponibles
python scripts/validate_results.py --list-materials

# Modifier les seuils de température
python scripts/validate_results.py --T-max-realistic 3500 --T-warning 2500
```

### Matériaux disponibles

- **Aluminum** - Aluminium structurel (T_max: 700K)
- **Carbon-Carbon** - Composite Space Shuttle (T_max: 3000K) [par défaut]
- **PICA-X** - Bouclier SpaceX Dragon (T_max: 2600K)
- **RCC** - Reinforced Carbon-Carbon Shuttle (T_max: 3300K)
- **Phenolic** - Résine phénolique Apollo (T_max: 2000K)
- **Stainless-Steel** - Acier inoxydable 304 (T_max: 1200K)

## Critères de validation

L'outil effectue 7 types de vérifications:

### 1. Limites de température matériau
- **CRITICAL** si T > T_max_service
- **CRITICAL** si T > T_melting
- **WARNING** si T > 2000K

### 2. Efficacité thermique (T_matériau / T_récupération)
- **CRITICAL** si efficacité > 0.95 (température trop proche de la récupération)
- **WARNING** si efficacité > 0.80
- Attendu: 0.2-0.6 pour TPS passif

### 3. Refroidissement radiatif manquant
- **ERROR** si q_rad/q_conv > 0.5 (radiation dominante)
- **WARNING** si q_rad/q_conv > 0.1
- Recommandation: Ajouter terme Stefan-Boltzmann au BC Robin

### 4. Flux thermique au point de stagnation
- **ERROR** si q > 10 MW/m² (limite pratique)
- **WARNING** si q > 5 MW/m²

### 5. Besoin d'ablation
- **ERROR** si T > T_ablation et pas de modèle d'ablation

### 6. Régime d'écoulement
- **WARNING** si Re hors plage [10⁴, 10⁸]
- **WARNING** si Mach > 5 (régime hypersonique)

### 7. Comparaison aux données de référence
- **WARNING** si écart > 50% vs missions historiques (Shuttle, Apollo)

## Format du rapport

Le rapport généré contient:

```
THERMAL ANALYSIS VALIDATION REPORT
================================================================================
Material: Carbon-Carbon (Space Shuttle nose)
T_max_service: 3000.0 K

SUMMARY
  PASSED:   X
  WARNING:  X
  ERROR:    X
  CRITICAL: X

CRITICAL ISSUES
[CRITICAL] Temperature exceeds material limit
  V = 5000 m/s: T_max = 11300 K > T_service = 3000 K
  -> Material failure imminent. Consider advanced TPS or active cooling.

RECOMMENDATIONS
1. IMMEDIATE: Add radiative cooling term
2. HIGH: Implement ablative TPS model
3. MEDIUM: Add real gas effects for V > 3000 m/s
```

## Export CSV enrichi

Le CSV exporté contient:

| Colonne | Description |
|---------|-------------|
| Velocity_m_s | Vitesse [m/s] |
| T_max_K | Température maximum [K] |
| T_recovery_K | Température de récupération [K] |
| Efficiency | Ratio T_max/T_recovery |
| Severity_Max | Sévérité maximum (CRITICAL, ERROR, WARNING, PASSED) |
| Issue_Count | Nombre de problèmes détectés |
| Issues | Liste des problèmes |

## Utilisation programmatique

```python
from src.validation import ThermalValidator, get_material

# Créer le validateur
validator = ThermalValidator(material=get_material("PICA-X"))

# Charger et valider
velocities, temperatures = validator.load_csv_results("path/to/results.csv")
results = validator.validate_complete_study("path/to/results.csv")

# Générer le rapport
report = validator.generate_report(results)
print(report)

# Exporter CSV enrichi
validator.export_validation_csv(
    velocities, temperatures, results,
    "path/to/flagged_results.csv"
)
```

## Interprétation des résultats

### Problème identifié: Efficacité = 1.0

Si vous voyez "Efficiency = 1.00" pour tous les cas:
- **Cause**: Le modèle FEM n'inclut pas le refroidissement radiatif
- **Conséquence**: La température du matériau atteint la température de récupération théorique
- **Solution**: Ajouter le terme de radiation Stefan-Boltzmann au boundary condition Robin

### Températures extrêmes (> 20,000K)

- Ce sont les **températures de récupération** (u_E dans le BC Robin), pas les températures matériau
- Physiquement correctes pour les vitesses orbitales (V > 7000 m/s)
- La température réelle du matériau devrait être 20-40% de cette valeur avec refroidissement radiatif

### Recommandations prioritaires

1. **IMMEDIAT**: Ajouter refroidissement radiatif
   - Modifier `src/core/assembly.py` ligne ~117
   - BC Robin: `-k*dT/dn = alpha*(T-T_E) - epsilon*sigma*T^4`

2. **HAUTE PRIORITE**: Modèle d'ablation pour T > 2000K
   - Refroidissement par pyrolyse
   - Perte de masse
   - Transpiration des gaz d'ablation

3. **MOYENNE**: Effets gaz réel pour M > 5
   - Dissociation chimique (O₂→2O, N₂→2N)
   - Propriétés variables (c_p, gamma)

## Codes de sortie

- **0**: Validation réussie (warnings possibles)
- **1**: Erreurs détectées (review recommandé)
- **2**: Problèmes critiques (attention immédiate requise)

## Support

Pour des questions ou rapporter des bugs, consulter le fichier principal [CLAUDE.md](../../CLAUDE.md) du projet.
