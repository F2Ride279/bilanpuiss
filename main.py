import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Configuration pour un affichage plus agréable des graphiques
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. Importation des données ---
def charger_donnees_zones(fichier_csv_zones):
    """Charge les données des zones à partir d'un fichier CSV."""
    try:
        df_zones = pd.read_csv(fichier_csv_zones)
        # S'assurer que les noms de colonnes sont bien ceux attendus (robustesse aux espaces)
        df_zones.columns = [col.strip() for col in df_zones.columns]
        print("Données des zones chargées avec succès.")
        print(df_zones.head())
        return df_zones
    except FileNotFoundError:
        print(f"Erreur : Le fichier {fichier_csv_zones} est introuvable.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données des zones : {e}")
        return None

def charger_donnees_liens(fichier_csv_liens):
    """Charge les données des liens interzones à partir d'un fichier CSV."""
    try:
        df_liens = pd.read_csv(fichier_csv_liens)
        df_liens.columns = [col.strip() for col in df_liens.columns]
        print("\nDonnées des liens chargées avec succès.")
        print(df_liens.head())
        return df_liens
    except FileNotFoundError:
        print(f"Erreur : Le fichier {fichier_csv_liens} est introuvable.")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données des liens : {e}")
        return None

# --- 2. Représentation graphique du réseau ---
def representer_reseau(df_zones, df_liens):
    """Représente graphiquement les zones et les liens interzones."""
    if df_zones is None or df_liens is None:
        print("Impossible de représenter le réseau : données manquantes.")
        return

    G = nx.Graph()
    positions = {}
    labels_nodes = {}

    for index, row in df_zones.iterrows():
        node_id = row['Numero de la zone']
        G.add_node(node_id)
        positions[node_id] = (row['Coordonnee en x'], row['Coordonnee en y'])
        labels_nodes[node_id] = f"Z{node_id}"

    for index, row in df_liens.iterrows():
        G.add_edge(row['Zone 1'], row['Zone 2'], capacity=row['Capacite du lien'])

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos=positions, with_labels=True, labels=labels_nodes,
            node_size=2000, node_color='skyblue', font_size=10, font_weight='bold',
            width=[d['capacity']/20 for u,v,d in G.edges(data=True)]) # Épaisseur des liens proportionnelle à la capacité

    labels_edges = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=labels_edges, font_color='red')

    plt.title("Représentation graphique du réseau de zones")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.grid(True)
    plt.show()

# --- 3. Calcul stochastique et bilan de puissance (Monte Carlo) ---
def simulation_monte_carlo_bilan(df_zones, nombre_simulations, facteur_std_dev=0.1):
    """
    Effectue une simulation de Monte Carlo pour le bilan de puissance.

    Args:
        df_zones (pd.DataFrame): DataFrame contenant les données des zones.
        nombre_simulations (int): Nombre d'itérations pour la simulation (ex: 12 pour mois, 8760 pour heures).
        facteur_std_dev (float): Facteur pour déterminer l'écart-type en % de la moyenne.
                                 Par exemple, 0.1 signifie un écart-type de 10% de la valeur de base.
    Returns:
        dict: Un dictionnaire où les clés sont les numéros de zone et les valeurs
              sont des listes des bilans de puissance nets pour chaque simulation.
    """
    if df_zones is None:
        print("Impossible de lancer la simulation : données des zones manquantes.")
        return None

    resultats_simulation = {zone_id: [] for zone_id in df_zones['Numero de la zone']}

    colonnes_a_varier = {
        'charge locale': 'charge_sim',
        'export de la zone': 'export_sim', # Peut être négatif (import)
        'production pilotable integree': 'prod_pilot_sim',
        'production variable integree': 'prod_var_sim'
    }

    for i in range(nombre_simulations):
        if (i + 1) % (nombre_simulations // 10 if nombre_simulations >=10 else 1) == 0 : # Affiche la progression
             print(f"Simulation Monte Carlo: Itération {i+1}/{nombre_simulations}")

        for index, row in df_zones.iterrows():
            zone_id = row['Numero de la zone']
            
            # Génération stochastique des valeurs
            # On utilise une distribution normale. Moyenne = valeur de base, Écart-type = X% de la moyenne.
            # Pour la charge et la production, on s'assure qu'elles ne sont pas négatives.
            charge_sim = max(0, np.random.normal(row['charge locale'], 
                                                 abs(row['charge locale'] * facteur_std_dev)))
            
            # L'export peut être négatif (import), donc pas de max(0, ...)
            export_sim = np.random.normal(row['export de la zone'], 
                                          abs(row['export de la zone'] * facteur_std_dev) if row['export de la zone'] != 0 else facteur_std_dev) # Eviter std=0 si export=0

            prod_pilot_sim = max(0, np.random.normal(row['production pilotable integree'], 
                                                     abs(row['production pilotable integree'] * facteur_std_dev)))
            
            prod_var_sim = max(0, np.random.normal(row['production variable integree'], 
                                                   abs(row['production variable integree'] * facteur_std_dev)))

            # Calcul du bilan de puissance net pour la zone
            # Bilan Net = (Total Production) - (Total Consommation Locale) - (Export Net)
            # Un export positif diminue le bilan de la zone. Un import (export négatif) l'augmente.
            bilan_net = (prod_pilot_sim + prod_var_sim) - charge_sim - export_sim
            
            resultats_simulation[zone_id].append(bilan_net)
            
    print("Simulation Monte Carlo terminée.")
    return resultats_simulation

def analyser_resultats_simulation(resultats_mc, df_zones):
    """Analyse et affiche les résultats de la simulation Monte Carlo."""
    if resultats_mc is None:
        print("Aucun résultat de simulation à analyser.")
        return

    print("\n--- Analyse des Résultats de la Simulation Monte Carlo ---")
    for zone_id, bilans in resultats_mc.items():
        zone_info = df_zones[df_zones['Numero de la zone'] == zone_id].iloc[0]
        
        moyenne_bilan = np.mean(bilans)
        std_dev_bilan = np.std(bilans)
        p5_bilan = np.percentile(bilans, 5)  # 5ème percentile (P5)
        p95_bilan = np.percentile(bilans, 95) # 95ème percentile (P95)
        prob_deficit = (np.sum(np.array(bilans) < 0) / len(bilans)) * 100 # Probabilité de bilan net négatif

        print(f"\nZone {zone_id}:")
        print(f"  Valeurs de base:")
        print(f"    Charge: {zone_info['charge locale']:.2f}, Export: {zone_info['export de la zone']:.2f}, "
              f"Prod. Pilot.: {zone_info['production pilotable integree']:.2f}, Prod. Var.: {zone_info['production variable integree']:.2f}")
        
        print(f"  Résultats stochastiques du bilan de puissance net:")
        print(f"    Moyenne: {moyenne_bilan:.2f}")
        print(f"    Écart-type: {std_dev_bilan:.2f}")
        print(f"    P5 (pire 5% des cas): {p5_bilan:.2f}")
        print(f"    P95 (meilleurs 5% des cas): {p95_bilan:.2f}")
        print(f"    Probabilité de déficit (bilan < 0): {prob_deficit:.2f}%")

        # Optionnel: Afficher un histogramme pour chaque zone
        plt.figure(figsize=(8, 5))
        plt.hist(bilans, bins=50, density=True, alpha=0.7, color='coral')
        plt.title(f"Distribution du Bilan de Puissance Net - Zone {zone_id}")
        plt.xlabel("Bilan de Puissance Net (MW ou unité)")
        plt.ylabel("Densité de probabilité")
        plt.axvline(moyenne_bilan, color='k', linestyle='dashed', linewidth=1, label=f'Moyenne: {moyenne_bilan:.2f}')
        plt.axvline(p5_bilan, color='r', linestyle='dotted', linewidth=1, label=f'P5: {p5_bilan:.2f}')
        plt.axvline(p95_bilan, color='g', linestyle='dotted', linewidth=1, label=f'P95: {p95_bilan:.2f}')
        plt.legend()
        plt.show()

# --- Programme Principal ---
if __name__ == "__main__":
    fichier_zones_csv = "zones.csv"
    fichier_liens_csv = "liens.csv"

    # 1. Charger les données
    df_zones = charger_donnees_zones(fichier_zones_csv)
    df_liens = charger_donnees_liens(fichier_liens_csv)

    if df_zones is not None and df_liens is not None:
        # 2. Représenter le réseau
        representer_reseau(df_zones, df_liens)

        # 3. Lancer la simulation Monte Carlo
        # Choix du nombre de simulations: 12 pour une analyse "mensuelle", 8760 pour une analyse "horaire"
        # Note: "mensuelle" ou "horaire" ici se réfère au nombre d'échantillons stochastiques.
        # Cela ne simule pas une dépendance temporelle entre les heures/mois, mais plutôt
        # le nombre de scénarios aléatoires générés à partir des valeurs de base.
        
        # Exemple avec 1000 simulations pour une bonne distribution
        nombre_iterations_mc = 1000 # Ou 12, ou 8760 selon l'interprétation souhaitée
        # Facteur d'écart-type: 10% de la moyenne pour cet exemple.
        # Augmenter pour plus de variabilité, diminuer pour moins.
        facteur_ecart_type = 0.2 # 20% de variabilité
        
        print(f"\nLancement de la simulation Monte Carlo avec {nombre_iterations_mc} itérations et un facteur d'écart-type de {facteur_ecart_type*100}%.")
        resultats_mc = simulation_monte_carlo_bilan(df_zones, nombre_iterations_mc, facteur_std_dev=facteur_ecart_type)

        # 4. Analyser et afficher les résultats
        if resultats_mc:
            analyser_resultats_simulation(resultats_mc, df_zones)
    else:
        print("Arrêt du programme en raison d'erreurs de chargement de données.")
