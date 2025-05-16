import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. Importation des données depuis un fichier Excel ---
def charger_donnees_zones_excel(fichier_excel, nom_onglet):
    """Charge les données des zones à partir d'un onglet d'un fichier Excel."""
    try:
        df_zones = pd.read_excel(fichier_excel, sheet_name=nom_onglet)
        df_zones.columns = [col.strip() for col in df_zones.columns]
        print(f"Données des zones chargées avec succès depuis l'onglet '{nom_onglet}' du fichier '{fichier_excel}'.")
        print(df_zones.head())
        return df_zones
    except FileNotFoundError:
        print(f"Erreur : Le fichier Excel '{fichier_excel}' est introuvable.")
        return None
    except ValueError as ve: # Peut se produire si l'onglet n'existe pas
        print(f"Erreur lors de la lecture de l'onglet '{nom_onglet}' : {ve}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données des zones depuis Excel : {e}")
        return None

def charger_donnees_liens_excel(fichier_excel, nom_onglet):
    """Charge les données des liens interzones à partir d'un onglet d'un fichier Excel."""
    try:
        df_liens = pd.read_excel(fichier_excel, sheet_name=nom_onglet)
        df_liens.columns = [col.strip() for col in df_liens.columns]
        print(f"\nDonnées des liens chargées avec succès depuis l'onglet '{nom_onglet}' du fichier '{fichier_excel}'.")
        print(df_liens.head())
        return df_liens
    except FileNotFoundError:
        print(f"Erreur : Le fichier Excel '{fichier_excel}' est introuvable.")
        return None
    except ValueError as ve: # Peut se produire si l'onglet n'existe pas
        print(f"Erreur lors de la lecture de l'onglet '{nom_onglet}' : {ve}")
        return None
    except Exception as e:
        print(f"Erreur lors du chargement des données des liens depuis Excel : {e}")
        return None

# --- 2. Représentation graphique du réseau (inchangée) ---
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
            width=[d['capacity']/20 for u,v,d in G.edges(data=True)])

    labels_edges = nx.get_edge_attributes(G, 'capacity')
    nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=labels_edges, font_color='red')

    plt.title("Représentation graphique du réseau de zones")
    plt.xlabel("Coordonnée X")
    plt.ylabel("Coordonnée Y")
    plt.grid(True)
    plt.show()

# --- 3. Calcul stochastique et bilan de puissance (Monte Carlo) (inchangée) ---
def simulation_monte_carlo_bilan(df_zones, nombre_simulations, facteur_std_dev=0.1):
    if df_zones is None:
        print("Impossible de lancer la simulation : données des zones manquantes.")
        return None
    resultats_simulation = {zone_id: [] for zone_id in df_zones['Numero de la zone']}
    for i in range(nombre_simulations):
        if (i + 1) % (nombre_simulations // 10 if nombre_simulations >=10 else 1) == 0:
             print(f"Simulation Monte Carlo: Itération {i+1}/{nombre_simulations}")
        for index, row in df_zones.iterrows():
            zone_id = row['Numero de la zone']
            charge_sim = max(0, np.random.normal(row['charge locale'], 
                                                 abs(row['charge locale'] * facteur_std_dev)))
            export_sim = np.random.normal(row['export de la zone'], 
                                          abs(row['export de la zone'] * facteur_std_dev) if row['export de la zone'] != 0 else facteur_std_dev)
            prod_pilot_sim = max(0, np.random.normal(row['production pilotable integree'], 
                                                     abs(row['production pilotable integree'] * facteur_std_dev)))
            prod_var_sim = max(0, np.random.normal(row['production variable integree'], 
                                                   abs(row['production variable integree'] * facteur_std_dev)))
            bilan_net = (prod_pilot_sim + prod_var_sim) - charge_sim - export_sim
            resultats_simulation[zone_id].append(bilan_net)
    print("Simulation Monte Carlo terminée.")
    return resultats_simulation

# --- 4. Analyser et afficher les résultats (inchangée) ---
def analyser_resultats_simulation(resultats_mc, df_zones):
    if resultats_mc is None:
        print("Aucun résultat de simulation à analyser.")
        return
    print("\n--- Analyse des Résultats de la Simulation Monte Carlo ---")
    for zone_id, bilans in resultats_mc.items():
        zone_info = df_zones[df_zones['Numero de la zone'] == zone_id].iloc[0]
        moyenne_bilan = np.mean(bilans)
        std_dev_bilan = np.std(bilans)
        p5_bilan = np.percentile(bilans, 5)
        p95_bilan = np.percentile(bilans, 95)
        prob_deficit = (np.sum(np.array(bilans) < 0) / len(bilans)) * 100
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
        plt.figure(figsize=(8, 5))
        plt.hist(bilans, bins=50, density=True, alpha=0.7, color='coral', edgecolor='black')
        plt.title(f"Distribution du Bilan de Puissance Net - Zone {zone_id}")
        plt.xlabel("Bilan de Puissance Net (MW ou unité)")
        plt.ylabel("Densité de probabilité")
        plt.axvline(moyenne_bilan, color='k', linestyle='dashed', linewidth=1, label=f'Moyenne: {moyenne_bilan:.2f}')
        plt.axvline(p5_bilan, color='r', linestyle='dotted', linewidth=1, label=f'P5: {p5_bilan:.2f}')
        plt.axvline(p95_bilan, color='g', linestyle='dotted', linewidth=1, label=f'P95: {p95_bilan:.2f}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

# --- Programme Principal ---
if __name__ == "__main__":
    fichier_excel_reseau = "reseau.xls" # Nom du fichier Excel
    nom_onglet_zones = "zones"            # Nom de l'onglet pour les données des zones
    nom_onglet_liens = "liens"            # Nom de l'onglet pour les données des liens

    # 1. Charger les données depuis le fichier Excel
    df_zones = charger_donnees_zones_excel(fichier_excel_reseau, nom_onglet_zones)
    df_liens = charger_donnees_liens_excel(fichier_excel_reseau, nom_onglet_liens)

    if df_zones is not None and df_liens is not None:
        # 2. Représenter le réseau
        representer_reseau(df_zones, df_liens)

        # 3. Lancer la simulation Monte Carlo
        nombre_iterations_mc = 1000 
        facteur_ecart_type = 0.2 
        
        print(f"\nLancement de la simulation Monte Carlo avec {nombre_iterations_mc} itérations et un facteur d'écart-type de {facteur_ecart_type*100}%.")
        resultats_mc = simulation_monte_carlo_bilan(df_zones, nombre_iterations_mc, facteur_std_dev=facteur_ecart_type)

        # 4. Analyser et afficher les résultats
        if resultats_mc:
            analyser_resultats_simulation(resultats_mc, df_zones)
    else:
        print("Arrêt du programme en raison d'erreurs de chargement de données.")
