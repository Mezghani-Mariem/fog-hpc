# Fog Node — Worker HPC

Service FastAPI identique sur les trois laptops physiques (`fog-1`, `fog-2`, `fog-3`).
Chaque laptop exécute **la même image Docker** avec un `NODE_ID` différent.

## Rôle

Le Fog Node est le **worker** du système FOG-HPC. Il :

- Exécute des jobs HPC indépendants (Monte Carlo, multiplication de matrices).
- Annonce son état au Scheduler (`/status`, `/health`).
- Expose des métriques Prometheus (`/metrics`).
- Rejette les jobs invalides ou qui dépassent sa capacité.

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | Vérifie que le service répond. |
| GET | `/status` | Disponibilité, jobs actifs, capacité, CPU/RAM. |
| POST | `/jobs` | Soumet un job (`monte_carlo` ou `matrix`). |
| GET | `/jobs/{job_id}` | Consulte un job par son identifiant. |
| GET | `/metrics` | Métriques Prometheus. |

## Workloads supportés

### Monte Carlo (`kind: "monte_carlo"`)

Estimation de π par tirages aléatoires dans un carré unité.

- Paramètre `size` : nombre d'échantillons (max **2 000 000**).
- Paramètre `seed` : graine aléatoire (par défaut `42`).
- Retour : `{"pi_estimate": 3.14...}`.

### Matrix (`kind: "matrix"`)

Multiplication de deux matrices carrées générées aléatoirement.

- Paramètre `size` : dimension (max **700**).
- Paramètre `seed` : graine aléatoire (par défaut `42`).
- Retour : `{"checksum": ..., "dimension": ...}`.

> On ne renvoie **pas** la matrice complète sur le réseau, seulement son checksum numérique.

## Exemple de requête

```json
POST /jobs
{
  "kind": "monte_carlo",
  "size": 100000,
  "seed": 42
}
```

Réponse :

```json
{
  "job_id": "e8b50207-5352-421c-930c-89659fabc260",
  "status": "completed",
  "node_id": "fog-1",
  "result": {"pi_estimate": 3.15008},
  "execution_seconds": 0.0079
}
```

## Variables d'environnement

| Variable | Description | Valeur par défaut |
|----------|-------------|-------------------|
| `NODE_ID` | Identifiant unique du nœud (`fog-1`, `fog-2`, `fog-3`). | `fog-1` |
| `MAX_RUNNING` | Nombre maximum de jobs simultanés. | `2` |

## Lancement local (sans Docker)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

set NODE_ID=fog-1
set MAX_RUNNING=2
python -m uvicorn fog_node.main:app --host 0.0.0.0 --port 8001
```

Puis ouvrir <http://127.0.0.1:8001/docs> pour tester via Swagger UI.

## Lancement avec Docker

Construire l'image :

```cmd
docker build -f Dockerfile.worker -t fog-hpc-node .
```

Lancer un conteneur :

```cmd
docker run --rm --name fog-node -p 8001:8001 -e NODE_ID=fog-1 -e MAX_RUNNING=2 fog-hpc-node
```

Simuler 3 nœuds sur une seule machine (tests) :

```cmd
docker run --rm --name fog-node-1 -p 8001:8001 -e NODE_ID=fog-1 -e MAX_RUNNING=2 fog-hpc-node
docker run --rm --name fog-node-2 -p 8002:8001 -e NODE_ID=fog-2 -e MAX_RUNNING=2 fog-hpc-node
docker run --rm --name fog-node-3 -p 8003:8001 -e NODE_ID=fog-3 -e MAX_RUNNING=2 fog-hpc-node
```

## Tests unitaires

```cmd
python -m pytest -q
```

Résultat attendu : 5 tests passent (health, monte_carlo, matrix, kind invalide, matrix trop grande).

## Métriques Prometheus exposées

- `foghpc_worker_jobs_total{node, outcome}` — Compteur des jobs par résultat.
- `foghpc_worker_running_jobs{node}` — Nombre de jobs en cours.
- `foghpc_worker_job_duration_seconds{node, kind}` — Histogramme des durées.

## Limites connues

- Pas de file persistante : les jobs en mémoire sont perdus au redémarrage.
- Pas de migration d'un job en cours vers un autre nœud.
- `cpu_percent` et `ram_percent` mesurés via `psutil` dans un conteneur ne représentent pas fidèlement la consommation isolée du processus.
- Timeout réseau : le calcul peut continuer côté worker même après expiration de la requête.

## Auteur

Rimeh ABIDI — Responsable Fog Node et workloads HPC.