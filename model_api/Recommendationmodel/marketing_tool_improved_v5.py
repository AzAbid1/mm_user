import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from transformers import DistilBertTokenizer, DistilBertModel, MarianMTModel, MarianTokenizer
import torch
import pickle
import os
from typing import Dict, List, Optional
import chromadb

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)

# Constantes
CATEGORIES = ['food', 'fashion', 'beauty', 'tech', 'lifestyle']
TONES = ['anticipatory', 'luxury', 'gourmand', 'authentique', 'inspirational', 'playful', 'professional']
PLATFORMS = ['instagram', 'facebook', 'tiktok', 'linkedin']
EMOTIONS = ['fierté', 'désir', 'douceur', 'joie', 'curiosité']
COPY_PATTERNS = ['AIDA', 'PAS', 'Storytelling']
LANGUAGES = ['français', 'anglais', 'arabe']

# Dictionnaire de traductions
TRANSLATIONS = {
    'français': {
        'error': 'Erreur',
        'adapted_to': 'Adapté à',
        'optimal_for': 'Optimal pour',
        'no_holiday': 'Aucun'
    },
    'anglais': {
        'error': 'Error',
        'adapted_to': 'Adapted to',
        'optimal_for': 'Optimal for',
        'no_holiday': 'None'
    },
    'arabe': {
        'error': 'خطأ',
        'adapted_to': 'مناسب لـ',
        'optimal_for': 'مثالي لـ',
        'no_holiday': 'لا يوجد'
    }
}

# Stratégies par catégorie
CATEGORY_STRATEGIES = {
    'food': [
        {'text': 'Reels recette avec {product}', 'score': 0.8},
        {'text': 'Défi dégustation #{product}', 'score': 0.7},
        {'text': 'Partenariat café local', 'score': 0.6},
        {'text': 'Foodie meetup Tunis', 'score': 0.5},
        {'text': 'Promo Ramadan pour {product}', 'score': 0.6}
    ],
    'fashion': [
        {'text': 'Reels styling {product}', 'score': 0.9},
        {'text': 'Défi look #{product}', 'score': 0.8},
        {'text': 'Pop-up shop Carthage', 'score': 0.7},
        {'text': 'Collab créateur local', 'score': 0.6},
        {'text': 'Live défilé Instagram', 'score': 0.5}
    ],
    'beauty': [
        {'text': 'Reels tuto {product}', 'score': 0.8},
        {'text': 'Quiz Stories teinte', 'score': 0.7},
        {'text': 'Défi beauté #{product}', 'score': 0.6},
        {'text': 'Giveaway 100 acheteurs', 'score': 0.5},
        {'text': 'Atelier makeup Tunis', 'score': 0.5}
    ],
    'tech': [
        {'text': 'Reels unboxing {product}', 'score': 0.9},
        {'text': 'Tournoi gaming', 'score': 0.8},
        {'text': 'Live Q&A tech', 'score': 0.7},
        {'text': 'Demo magasin Tunis', 'score': 0.6},
        {'text': 'Promo précommande', 'score': 0.5}
    ],
    'lifestyle': [
        {'text': 'Posts culturels {product}', 'score': 0.7},
        {'text': 'Vlog intégration {product}', 'score': 0.6},
        {'text': 'Marché local Tunis', 'score': 0.5},
        {'text': 'Sondage Stories', 'score': 0.5},
        {'text': 'Podcast sponsorisé', 'score': 0.4}
    ]
}

# Météo simulée
WEATHER_CONDITIONS = {
    'Tunis': {'temp': np.random.randint(15, 35), 'condition': np.random.choice(['ensoleillé', 'nuageux', 'pluvieux'])}
}

# Historique des performances
PERFORMANCE_HISTORY_FILE = "performance_history.pkl"
def load_performance_history():
    logging.info("DEBUG: Chargement de l'historique des performances")
    if os.path.exists(PERFORMANCE_HISTORY_FILE):
        try:
            with open(PERFORMANCE_HISTORY_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'historique : {str(e)}")
            return []
    return []

def save_performance_history(history):
    logging.info("DEBUG: Sauvegarde de l'historique des performances")
    try:
        with open(PERFORMANCE_HISTORY_FILE, 'wb') as f:
            pickle.dump(history, f)
    except Exception as e:
        logging.error(f"Erreur lors de la sauvegarde de l'historique : {str(e)}")

PERFORMANCE_HISTORY = load_performance_history()

# Cache des modèles
MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Fonction de traduction
def translate_text(text: str, target_lang: str) -> str:
    logging.info(f"DEBUG: Traduction du texte '{text}' vers {target_lang}")
    try:
        if target_lang == 'français':
            return text
        model_name = {
            'anglais': 'Helsinki-NLP/opus-mt-fr-en',
            'arabe': 'Helsinki-NLP/opus-mt-fr-ar'
        }.get(target_lang)
        if not model_name:
            logging.warning(f"DEBUG: Langue {target_lang} non supportée")
            return text
        tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        model = MarianMTModel.from_pretrained(model_name, cache_dir=MODEL_CACHE_DIR)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        logging.info(f"DEBUG: Traduction réussie : {result}")
        return result
    except Exception as e:
        logging.error(f"Erreur dans translate_text : {str(e)}")
        return text

# Générer texte basé sur CATEGORY_STRATEGIES
def generate_text_from_strategies(product: str, category: str, emotion: str, max_length: int = 100) -> str:
    logging.info(f"DEBUG: Génération texte pour {product}, {category}, {emotion}")
    try:
        strategies = CATEGORY_STRATEGIES.get(category, CATEGORY_STRATEGIES['lifestyle'])
        selected_strategy = max(strategies, key=lambda x: x['score'])['text'].format(product=product)
        text = f"{selected_strategy} : {emotion} au top !"
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        logging.info(f"DEBUG: Texte généré : {text}")
        return text
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_text_from_strategies : {str(e)}")
        return f"Shine avec {product}, vibe {emotion} !"

# Jours fériés tunisiens 2025
HOLIDAYS_TN_2025 = {
    "2025-01-01": {"name": "Jour de l'An", "theme": "nouvel an"},
    "2025-03-20": {"name": "Fête de l'Indépendance", "theme": "indépendance"},
    "2025-04-09": {"name": "Journée des Martyrs", "theme": "martyrs"},
    "2025-05-01": {"name": "Fête du Travail", "theme": "travail"},
    "2025-06-01": {"name": "Aïd el-Fitr", "theme": "aïd el-fitr"},
    "2025-08-07": {"name": "Aïd el-Adha", "theme": "aïd el-adha"},
    "2025-07-25": {"name": "Fête de la République", "theme": "république"},
    "2025-08-13": {"name": "Journée de la Femme", "theme": "femme"},
    "2025-10-15": {"name": "Fête de l'Évacuation", "theme": "évacuation"},
    "2025-08-28": {"name": "Ras el-Am el-Hijri", "theme": "nouvel an hijri"}
}

def test_holiday_api(date: str, country: str = "TN") -> dict:
    logging.info(f"DEBUG: Vérification des jours fériés pour {date} en {country}")
    try:
        target_date = pd.to_datetime(date).strftime('%Y-%m-%d')
        holiday = HOLIDAYS_TN_2025.get(target_date)
        if holiday:
            logging.info(f"DEBUG: Jour férié trouvé : {holiday['name']}")
            return {"name": holiday['name'], "theme": holiday['theme']}
        logging.info("DEBUG: Aucun jour férié trouvé")
        return {"name": None, "theme": None}
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans test_holiday_api : {str(e)}")
        return {"name": None, "theme": None}

# Tendances Twitter/X (simulé)
def get_twitter_trends(woeid: str = "23424977", category: str = "beauty") -> List[str]:
    logging.info(f"DEBUG: Simulation des tendances Twitter pour catégorie {category}")
    return [f"#{category.capitalize()}Tunis", "#Tunisie2025"]

def get_current_event(date: pd.Timestamp) -> tuple:
    holiday = test_holiday_api(date.strftime('%Y-%m-%d'))
    return holiday['name'], holiday['theme'] if holiday['name'] else None

# Clustering des utilisateurs
def cluster_users(category: str) -> Dict:
    logging.info(f"DEBUG: Clustering des utilisateurs pour {category}")
    try:
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'age': np.random.randint(18, 65, n_samples),
            'income': np.random.randint(500, 5000, n_samples),
            'engagement': np.random.uniform(0.5, 10, n_samples),
            'category_interest': [category] * n_samples
        })

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data[['age', 'income', 'engagement']])
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['cluster'] = kmeans.fit_predict(X_scaled)

        clusters = {
            0: {'name': f"{category.capitalize()} Luxe", 'hours': [19, 20], 'hashtags': ['#TunisianElite', '#LuxeTunisien']},
            1: {'name': f"{category.capitalize()} Budget", 'hours': [12, 13], 'hashtags': ['#TunisianVibes', '#BonPlan']},
            2: {'name': f"{category.capitalize()} Casual", 'hours': [17, 18], 'hashtags': ['#TunisianStyle', '#DailyVibes']}
        }
        result = clusters[data['cluster'].mode()[0]]
        logging.info(f"DEBUG: Cluster sélectionné : {result['name']}")
        return result
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans cluster_users : {str(e)}")
        return {'name': f"{category.capitalize()} Budget", 'hours': [12, 13], 'hashtags': ['#TunisianVibes']}

# Générer des hashtags
def generate_hashtags(product: str, category: str, cluster: Dict, event: Optional[str] = None, lang: str = 'français') -> tuple:
    logging.info(f"DEBUG: Génération des hashtags pour {product}, {category}, event={event}, lang={lang}")
    try:
        base_hashtags = {
            'food': ['#TunisianFood', '#CuisineTunisienne', '#TunisianSweets'],
            'fashion': ['#BijouTunisien', '#ModeTunisienne', '#EléganceNaturelle'],
            'beauty': ['#TunisianBeauty', '#BeautéTunisienne'],
            'tech': ['#TechTunis', '#InnovationTunisienne', '#GamingTunis'],
            'lifestyle': ['#TunisianLifestyle', '#VieTunisienne']
        }
        hashtags = base_hashtags.get(category, ['#TunisianVibes']) + cluster['hashtags']
        if event:
            event_hashtags = {
                'Ramadan': ['#RamadanKareem', '#IftarVibes'],
                'Eid al-Fitr': ['#EidMubarak', '#TunisianEid'],
                'Summer Festival': ['#TunisianSummer', '#BeachVibes'],
                'Independence Day': ['#TunisianPride', '#IndependenceDay']
            }
            hashtags.extend(event_hashtags.get(event, []))
        hashtags.append(f"#{product.replace(' ', '')}")
        hashtags.extend(get_twitter_trends(category=category)[:2])
        if lang != 'français':
            hashtags = [translate_text(tag, lang) for tag in hashtags]
        analysis = [{'hashtag': tag, 'reach': np.random.randint(10000, 100000)} for tag in hashtags[:5]]
        logging.info(f"DEBUG: Hashtags générés : {hashtags[:5]}")
        return hashtags[:5], analysis
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_hashtags : {str(e)}")
        return ['#TunisianVibes'], [{'hashtag': '#TunisianVibes', 'reach': 10000}]

# Générer une description
def generate_post_description(product: str, category: str, tone: str, emotion: str, cluster_name: str, pattern: str, event: Optional[str] = None, lang: str = 'français') -> tuple:
    logging.info(f"DEBUG: Génération de la description pour {product}, {category}, {tone}, {emotion}")
    try:
        desc = generate_text_from_strategies(product, category, emotion, max_length=80)
        cta = f"Découvrez {product.lower()} !"
        if event:
            desc += f" pour {event} !"
        if lang != 'français':
            desc = translate_text(desc, lang)
            cta = translate_text(cta, lang)
        logging.info(f"DEBUG: Description générée : {desc} | CTA : {cta}")
        return desc, cta
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_post_description : {str(e)}")
        return f"{product} : {emotion} pour {event or 'tous'} !", f"Découvrez {product.lower()} !"

# Générer un mood board
def generate_mood_board(category: str, tone: str, emotion: str) -> Dict:
    logging.info(f"DEBUG: Génération du mood board pour {category}, {tone}, {emotion}")
    try:
        mood_boards = {
            'food': {'colors': ['orange', 'red'], 'textures': ['rustique', 'chaud'], 'vibes': 'gourmand'},
            'fashion': {'colors': ['gold', 'white'], 'textures': ['soyeux', 'artisanat'], 'vibes': 'élégant'},
            'beauty': {'colors': ['pink', 'pastel'], 'textures': ['lisse', 'brillant'], 'vibes': 'glow'},
            'tech': {'colors': ['blue', 'black'], 'textures': ['futuriste', 'métallique'], 'vibes': 'innovant'},
            'lifestyle': {'colors': ['green', 'beige'], 'textures': ['naturel', 'doux'], 'vibes': 'authentique'}
        }
        base = mood_boards.get(category, {'colors': ['neutral'], 'textures': ['simple'], 'vibes': 'générique'})
        if tone == 'luxury':
            base['colors'].append('gold')
        if emotion == 'fierté':
            base['vibes'] += ', patriotique'
        logging.info(f"DEBUG: Mood board généré : {base}")
        return base
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_mood_board : {str(e)}")
        return {'colors': ['neutral'], 'textures': ['simple'], 'vibes': 'générique'}

# Générer un visuel
def generate_ai_visual(product: str, category: str, tone: str, emotion: str, event: Optional[str] = None, lang: str = 'français') -> Dict:
    logging.info(f"DEBUG: Génération du visuel pour {product}, {category}, {tone}, {emotion}")
    try:
        mood = generate_mood_board(category, tone, emotion)
        desc = f"Visuel pour {product} ({category}) : ambiance {mood['vibes']}, couleurs {', '.join(mood['colors'])}, textures {', '.join(mood['textures'])}."
        if event:
            event_elements = {
                'Ramadan': ', table d’iftar, lanternes',
                'Eid al-Fitr': ', décor festif, douceurs',
                'Summer Festival': ', plage, soleil',
                'Independence Day': ', drapeau tunisien'
            }
            desc += event_elements.get(event, f', thème {event.lower()}')
        desc += f" Texte superposé : 'Vivez {emotion} avec {product} ✨'"
        if lang != 'français':
            desc = translate_text(desc, lang)
        logging.info(f"DEBUG: Visuel généré : {desc}")
        return {'visual_description': desc, 'mood_board': mood}
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_ai_visual : {str(e)}")
        return {'visual_description': f"Visuel pour {product}", 'mood_board': {}}

# Recommander le format
def recommend_format(category: str, platform: str, lang: str = 'français') -> Dict:
    logging.info(f"DEBUG: Recommandation du format pour {category}, {platform}")
    try:
        formats = {
            'instagram': {
                'food': {'type': 'Carrousel', 'desc': 'Photos de recettes + étapes'},
                'fashion': {'type': 'Reel', 'desc': 'Vidéo de styling avec musique tendance'},
                'beauty': {'type': 'Story', 'desc': 'Tutoriel rapide avec swipe-up'},
                'tech': {'type': 'Reel', 'desc': 'Unboxing ou démo performance'},
                'lifestyle': {'type': 'Carrousel', 'desc': 'Moments de vie avec produit'}
            },
            'facebook': {
                'food': {'type': 'Post', 'desc': 'Image appétissante + lien recette'},
                'fashion': {'type': 'Album', 'desc': 'Collection de looks'},
                'beauty': {'type': 'Video', 'desc': 'Tutoriel détaillé'},
                'tech': {'type': 'Post', 'desc': 'Visuel tech + specs'},
                'lifestyle': {'type': 'Post', 'desc': 'Histoire culturelle'}
            }
        }
        format_rec = formats.get(platform, formats['instagram']).get(category, {'type': 'Post', 'desc': 'Image générique'})
        if lang != 'français':
            format_rec = {
                'type': translate_text(format_rec['type'], lang),
                'desc': translate_text(format_rec['desc'], lang)
            }
        logging.info(f"DEBUG: Format recommandé : {format_rec}")
        return format_rec
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans recommend_format : {str(e)}")
        return {'type': 'Post', 'desc': 'Image générique'}

# Prédire l’engagement
def predict_engagement(category: str, tone: str, platform: str, hour: float, model: RandomForestRegressor, scaler: StandardScaler) -> Dict:
    logging.info(f"DEBUG: Prédiction de l'engagement pour {category}, {tone}, {platform}, hour={hour}")
    try:
        data = pd.DataFrame({
            'hour': [hour],
            'category': [CATEGORIES.index(category)],
            'tone': [TONES.index(tone)],
            'platform': [PLATFORMS.index(platform)],
            'hashtag_engagement': [np.random.uniform(1, 3)]
        })
        X = scaler.transform(data)
        engagement = model.predict(X)[0]
        result = {
            'predicted_engagement': round(engagement, 1),
            'reach': int(engagement * 10000),
            'impressions': int(engagement * 15000),
            'interactions': int(engagement * 500)
        }
        logging.info(f"DEBUG: Engagement prédit : {result}")
        return result
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans predict_engagement : {str(e)}")
        return {'predicted_engagement': 1.0, 'reach': 10000, 'impressions': 15000, 'interactions': 500}

# Feedback loop
def update_performance_history(product: str, category: str, strategy: str, engagement: float) -> None:
    logging.info(f"DEBUG: Mise à jour de l'historique pour {product}, {category}, {strategy}, engagement={engagement}")
    try:
        PERFORMANCE_HISTORY.append({
            'product': product,
            'category': category,
            'strategy': strategy,
            'engagement': engagement,
            'timestamp': datetime.now()
        })
        save_performance_history(PERFORMANCE_HISTORY)
        logging.info(f"DEBUG: Performance ajoutée : {strategy} -> {engagement}%")
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans update_performance_history : {str(e)}")

# Générer un plan hebdomadaire
def generate_weekly_plan(product: str, category: str, tone: str, platform: str, cluster: Dict, lang: str = 'français') -> List[Dict]:
    logging.info(f"DEBUG: Génération du plan hebdomadaire pour {product}, {category}, {tone}, {platform}")
    try:
        plan = []
        base_date = pd.Timestamp.now()
        formats = ['Reel', 'Story', 'Carrousel', 'Post']
        for i in range(7):
            date = base_date + timedelta(days=i)
            event, _ = get_current_event(date)
            format_type = np.random.choice(formats)
            desc, cta = generate_post_description(product, category, tone, np.random.choice(EMOTIONS), cluster['name'], np.random.choice(COPY_PATTERNS), event, lang)
            plan.append({
                'date': date.strftime('%Y-%m-%d'),
                'format': translate_text(format_type, lang) if lang != 'français' else format_type,
                'description': desc,
                'cta': cta,
                'hour': np.random.choice(cluster['hours']),
                'priority': translate_text('À publier' if i < 3 else 'À planifier', lang)
            })
        logging.info(f"DEBUG: Plan hebdomadaire généré : {len(plan)} entrées")
        return plan
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_weekly_plan : {str(e)}")
        return []

# Stratégie marketing
def get_tunisia_marketing_strategy(product: str, category: str, tone: str, cluster: Dict, emotion: str, event: Optional[str] = None, lang: str = 'français') -> Dict:
    logging.info(f"DEBUG: Génération de la stratégie pour {product}, {category}, {tone}, {emotion}")
    try:
        strategies = CATEGORY_STRATEGIES.get(category, CATEGORY_STRATEGIES['lifestyle'])
        for strat in strategies:
            strat['text'] = strat['text'].format(product=product)
            past_perf = [p['engagement'] for p in PERFORMANCE_HISTORY if p['strategy'] == strat['text']]
            if past_perf:
                strat['score'] *= (1 + np.mean(past_perf) / 100)
            if emotion == 'fierté':
                strat['score'] *= 1.1
            if cluster['name'].endswith('Luxe'):
                strat['score'] *= 1.2

        strategies = sorted(strategies, key=lambda x: x['score'], reverse=True)[:3]
        weather = WEATHER_CONDITIONS['Tunis']
        if weather['condition'] == 'pluvieux':
            strategies.append({'text': f"Promo pluie : -10% sur {product}", 'score': 0.6})

        strategy = {
            'campaign_name': f"Campagne {product}",
            'key_strategies': [s['text'] for s in strategies],
            'influenceur': f"Collab avec @{category}_influencer_tn",
            'promo': f"10-15% de réduction pour {cluster['name']}"
        }
        if lang != 'français':
            strategy = {
                'campaign_name': translate_text(strategy['campaign_name'], lang),
                'key_strategies': [translate_text(s, lang) for s in strategy['key_strategies']],
                'influenceur': translate_text(strategy['influenceur'], lang),
                'promo': translate_text(strategy['promo'], lang)
            }
        logging.info(f"DEBUG: Stratégie générée : {strategy}")
        return strategy
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans get_tunisia_marketing_strategy : {str(e)}")
        return {'campaign_name': f"Campagne {product}", 'key_strategies': [], 'influenceur': '', 'promo': ''}

# Initialiser Chroma
chroma_client = chromadb.Client()
collection_name = "recommendations"
try:
    collection = chroma_client.get_collection(collection_name)
except:
    collection = chroma_client.create_collection(collection_name)

# Générer un embedding
def generate_embedding(text: str) -> list:
    logging.info(f"DEBUG: Génération embedding pour texte : {text[:50]}...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=MODEL_CACHE_DIR)
        model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=MODEL_CACHE_DIR)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
        logging.info("DEBUG: Embedding généré")
        return embedding
    except Exception as e:
        logging.error(f"DEBUG: Erreur dans generate_embedding : {str(e)}")
        return [0.0] * 768  # Vecteur par défaut

# Sélectionner une tonalité statique
def select_tone(emotion: str) -> str:
    logging.info(f"DEBUG: Sélection de la tonalité pour émotion {emotion}")
    tone_mapping = {
        'fierté': 'inspirational',
        'désir': 'luxury',
        'douceur': 'authentique',
        'joie': 'playful',
        'curiosité': 'anticipatory'
    }
    tone = tone_mapping.get(emotion, 'authentique')
    logging.info(f"DEBUG: Tonalité sélectionnée : {tone}")
    return tone

# Convertir les types NumPy
def convert_numpy_types(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Fonction principale
def recommend_post_format(
    product: str,
    category: str,
    tone: str = 'auto',
    platform: str = 'instagram',
    emotion: str = 'joie',
    base_price: Optional[float] = None,
    date: Optional[pd.Timestamp] = None,
    lang: str = 'français'
) -> Dict:
    logging.info(f"DEBUG: Début de recommend_post_format pour {product}, {category}, {tone}, {platform}")
    if not date:
        date = pd.Timestamp.now()

    try:
        if not product or category not in CATEGORIES:
            raise ValueError("Produit vide ou catégorie invalide")

        cluster = cluster_users(category)
        if tone == 'auto':
            tone = select_tone(emotion)

        event, event_info = get_current_event(date)
        hashtags, hashtag_analysis = generate_hashtags(product, category, cluster, event, lang)
        desc, cta = generate_post_description(product, category, tone, emotion, cluster['name'], np.random.choice(COPY_PATTERNS), event, lang)
        visual = generate_ai_visual(product, category, tone, emotion, event, lang)
        format_rec = recommend_format(category, platform, lang)
        strategy = get_tunisia_marketing_strategy(product, category, tone, cluster, emotion, event, lang)

        n_samples = 100
        X = np.random.rand(n_samples, 5)
        y = np.random.uniform(0.5, 5, n_samples) + (X[:, 0] > 0.7) * 2
        model = RandomForestRegressor(random_state=42).fit(X, y)
        scaler = StandardScaler().fit(X)
        engagement = predict_engagement(category, tone, platform, date.hour, model, scaler)

        weekly_plan = generate_weekly_plan(product, category, tone, platform, cluster, lang)

        budget = np.random.uniform(50, 500)
        roi = engagement['predicted_engagement'] * 2

        coach = f"💬 Coach IA : Super choix avec {product} ! Publiez un {format_rec['type']} à {cluster['hours'][0]}h pour capter {cluster['name']}. Utilisez l’émotion {emotion} pour un max d’impact !"
        if lang != 'français':
            coach = translate_text(coach, lang)

        result = {
            'tone': {'primary': tone, 'justification': translate_text(f"Adapté à {cluster['name']} et {emotion}", lang)},
            'content': {'description': desc, 'cta': cta, 'format': format_rec},
            'visual': visual,
            'hashtags': {'tags': hashtags, 'analysis': hashtag_analysis},
            'posting_time': {'hour': cluster['hours'][0], 'justification': translate_text(f"Optimal pour {cluster['name']}", lang)},
            'strategy': strategy,
            'engagement': engagement,
            'weekly_plan': weekly_plan,
            'budget_roi': {'budget': round(budget, 2), 'roi': f"{round(roi, 1)}%"},
            'coach': coach
        }

        # Générer un embedding pour la description
        desc_embedding = generate_embedding(result['content']['description'])
        rec_id = f"{product}_{pd.Timestamp.now().timestamp()}"
        collection.add(
            embeddings=[desc_embedding],
            documents=[result['content']['description']],
            metadatas=[{
                "product": product,
                "category": category,
                "description": result['content']['description'],
                "hashtags": ", ".join(result['hashtags']['tags']),
                "timestamp": pd.Timestamp.now().isoformat()
            }],
            ids=[rec_id]
        )

        # Convertir les types NumPy
        result = convert_numpy_types(result)
        logging.info("DEBUG: Recommandations générées avec succès")
        return result

    except Exception as e:
        logging.error(f"DEBUG: Erreur dans recommend_post_format : {str(e)}")
        return {"error": translate_text(str(e), lang)}

if __name__ == "__main__":
    logging.info("DEBUG: Test de recommend_post_format")
    recs = recommend_post_format("Collier", "fashion", "authentique", "instagram", "joie", 45.0)
    print(recs)