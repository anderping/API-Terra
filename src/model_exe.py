import pandas as pd
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.class_weight import compute_class_weight

from sentence_transformers import SentenceTransformer

# from symspellpy.symspellpy import SymSpell, Verbosity
from googletrans import Translator


# Funciones para cargar y corregir texto en español:

# def cargar_symspell_es(diccionario_path='frequency_dictionary_es.txt'):
#     sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

#     if not sym_spell.load_dictionary(diccionario_path, term_index=0, count_index=1):
#         raise Exception("No se pudo cargar el diccionario español.")
    
#     return sym_spell


# def correct_text(text, sym_spell):
#     suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
#     return suggestions[0].term if suggestions else text


def translate(text):
    """Traduce el texto al inglés si no está en ese idioma."""

    # sym_spell_es = cargar_symspell_es()
    translator = Translator()

    if not text:
        return ""
    
    try:
        src_lan = translator.detect(text)
        lang = src_lan.lang

        # if lang == 'es':
        #     text = sym_spell_es.lookup_compound(text, max_edit_distance=2)[0].term
        #     print(f"PRIMERO: {text}")

        if lang != 'en':
            trans_text = translator.translate(text, src=lang, dest='en')

            return trans_text.text
        
        else:
            return text
    
    except Exception as e:
        print(f"Error en traducción: {e}")

        return text
    

PLACEHOLDERS = {
    "element": ["button", "form", "image", "text", "table", "menu", "link", "icon", "header", "footer"],
    "section": ["homepage", "landing_page", "contact_form", "sidebar", "modal", "checkout_page"],
    "action": ["click", "submit", "scroll", "hover", "load", "select", "toggle"],
    "device": ["mobile", "desktop", "tablet", "iOS", "Android"],
    "issue": ["broken", "slow", "overlapping", "invisible", "glitching", "misaligned"]
}

DOMAIN_TERMS = {
    "add": ["add", "include", "insert", "implement", "create", "introduce", "incorporate", "append", "attach", "embed", "integrate", "build in", "put in", "place", "install", "set up", "deploy", "provide", "offer", "develop", "design", "compose", "construct", "assemble", "intercalate", "interpose", "subjoin", "enlist", "affix", "adjoin"],
    "copy": ["copy", "text", "reword", "revise", "update", "edit", "rewrite", "polish", "amend", "proofread", "rephrase", "adjust", "clarify", "correct", "copyedit", "proof", "tidy", "simplify", "sanitize", "paraphrase", "restate"],
    "design": ["design", "layout", "align", "color", "format", "style", "typography", "spacing", "contrast", "font", "graphics", "composition", "redesign", "resize", "realign", "sharpen", "vectorize", "recolor", "mock"],
    "fix": ["bug", "fix", "error", "broken", "debug", "patch", "resolve", "correct", "repair", "troubleshoot", "solve", "mend", "crash", "malfunction", "glitch", "timeout", "freeze", "hang", "misfire"],
    "change": ["change", "modify", "implement", "alter", "adapt", "revamp", "rework", "refine", "optimize", "enhance", "remodel", "restructure", "realign", "streamline", "reprogram", "modernize", "fine-tune", "reengineer", "switch", "update"]
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocesador de texto para extraer verbos de acción, términos de dominio y marcadores de posición."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_action_verbs(self, text):
        doc = self.nlp(text)
        verbs = []
        for token in doc:
            if token.pos_ == "VERB" or (token.dep_ in ("xcomp", "ccomp") and token.head.pos_ == "VERB"):
                lemma = token.lemma_.lower()
                if lemma not in ["be", "have", "do", "make", "get"]:
                    phrase = " ".join([child.text.lower() for child in token.children 
                                     if child.dep_ in ("advmod", "dobj", "prt") and child.pos_ != "PUNCT"])
                    verbs.append(f"{lemma} {phrase}".strip())
        return list(set(verbs))

    def enhance_with_domain_terms(self, text, verbs):
        doc = self.nlp(text)
        for token in doc:
            for category, terms in DOMAIN_TERMS.items():
                if token.lemma_.lower() in terms:
                    verbs.append(f"{category}:{token.lemma_.lower()}")
        return verbs

    def extract_placeholders(self, text):
        tags = []
        lowered = text.lower()
        for category, terms in PLACEHOLDERS.items():
            for term in terms:
                if term.lower() in lowered:
                    tags.append(f"{category}_{term}")
        return tags
    
    def process_text(self, text):
        verbs = self.extract_action_verbs(text)
        verbs = self.enhance_with_domain_terms(text, verbs)
        placeholders = self.extract_placeholders(text)
        return " ".join(verbs + placeholders)
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X["request_text"].apply(self.process_text).tolist()
        return [self.process_text(text) for text in X]


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        embeddings = self.encoder.encode(X)
        self.scaler.fit(embeddings)
        return self
        
    def transform(self, X):
        embeddings = self.encoder.encode(X)
        return self.scaler.transform(embeddings)


def build_full_pipeline(n_estimators=200, model_name='all-MiniLM-L6-v2', valid_classes=None):
    
    # Definir clases válidas
    if valid_classes is None:
        valid_classes = ["Bug Fix", "Copy Revision", "Design Issues", "New Item", "Requested Change"]

    # Preprocesador de texto
    text_preprocessor = TextPreprocessor()
    
    # Pipeline para características de texto
    text_features = Pipeline([
        ('preprocessor', text_preprocessor),
        ('tfidf', TfidfVectorizer(
            max_features=2000,
            ngram_range=(2,4), 
            analyzer="char_wb",
            min_df=2
        ))
    ])
    
    # Pipeline para embeddings
    embedding_features = Pipeline([
        ('preprocessor', text_preprocessor),
        ('embedder', SentenceEmbedder(model_name=model_name))
    ])
    
    # Combinar características
    feature_union = FeatureUnion([
        ('text_features', text_features),
        ('embedding_features', embedding_features)
    ])


    # Pipeline completo con clasificador
    full_pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', XGBClassifier(
            n_estimators=n_estimators, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8, 
            random_state=42,
            eval_metric='mlogloss',
            num_class=len(valid_classes)
            ))
    ])
    
    return full_pipeline


class RequestClassifierPipeline:
    def __init__(self, n_estimators=100, contamination=0.1, model_name='all-MiniLM-L6-v2'):
        self.pipeline = build_full_pipeline(n_estimators, model_name)
        self.iso_forest = IsolationForest(contamination=contamination, random_state=42)
        self.label_encoder = LabelEncoder()
        self.contamination = contamination
    
    def _calculate_sample_weights(self, y_encoded):
        """Versión simplificada pero funcional"""
        """
        Calcula pesos para balanceo de clases usando la estrategia 'balanced' de scikit-learn.

        Args:
            y_encoded (array): Etiquetas codificadas numéricamente

        Returns:
            array: Pesos para cada muestra o None si ocurre un error
        """
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        return weights[y_encoded]  
    
    def fit(self, X, y):
        # Codificar etiquetas
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        sample_weights = self._calculate_sample_weights(y_encoded)
        
        fit_params = {'classifier__sample_weight': sample_weights} if sample_weights is not None else {}
        self.pipeline.fit(X, y_encoded, **fit_params)
        
        # Entrenar detector de anomalías
        features = self.pipeline.named_steps['features'].transform(X)
        self.iso_forest.fit(features)
        
        return self
        
    def predict(self, X, class_thresholds=None, anomaly_percentile=20, return_probs=False, uncertainty_handling='label'):
        """
        Realiza predicciones con umbrales personalizados por clase y manejo de incertidumbre.
        
        Parámetros:
        -----------
        X : pd.DataFrame o list
            Datos de entrada a predecir
        class_thresholds : dict, optional
            Diccionario con umbrales por clase (ej: {"Bug Fix": 0.9})
        anomaly_percentile : float, optional
            Percentil para detección de anomalías (default: 15)
        return_probs : bool, optional
            Si True, retorna solo las predicciones numéricas sin post-procesamiento
        uncertainty_handling : str, optional
            Estrategia para manejar predicciones inciertas ('label', 'original', 'majority')
        
        Retorna:
        --------
        pd.DataFrame
            DataFrame con predicciones y metadatos
        """
        # Umbrales
        if class_thresholds is None:
            class_thresholds = {
                "Bug Fix": 0.9,
                "Copy Revision": 0.75,
                "Design Issues": 0.92,
                "New Item": 0.95,
                "Requested Change": 0.93
            }

        probs = self.pipeline.predict_proba(X)
        y_pred = self.pipeline.predict(X)
        
        if return_probs:
            return y_pred
        
        # Convertir a array numpy para operaciones vectorizadas
        max_probs = probs.max(axis=1)
        pred_classes = probs.argmax(axis=1)
        
        # Verificar qué predicciones no superan el umbral de su clase
        thresholds_array = np.array([class_thresholds[cls] for cls in self.classes_])
        uncertain_mask = max_probs < thresholds_array[pred_classes]
        
        # Manejar predicciones inciertas según la estrategia seleccionada
        if uncertainty_handling == 'majority':
            # Asignar la clase mayoritaria cuando la confianza es baja
            majority_class = np.argmax(np.bincount(y_pred))
            y_pred[uncertain_mask] = majority_class

        elif uncertainty_handling == 'original' and isinstance(X, pd.DataFrame) and "type" in X.columns:
            y_pred[uncertain_mask] = self.label_encoder.transform(X["type"][uncertain_mask])

        elif uncertainty_handling == 'second_best_not_addressing' and isinstance(X, pd.DataFrame) and "type" in X.columns:
            # Calcular segunda mejor clase
            second_best_idx = probs.argsort(axis=1)[:, -2]

            for i in np.where(uncertain_mask)[0]:
                if X["type"].iloc[i] == "Not Addressing":
                    # Forzar segunda mejor
                    y_pred[i] = second_best_idx[i]
                else:
                    # Mantener etiqueta original
                    y_pred[i] = self.label_encoder.transform([X["type"].iloc[i]])[0]

        # Detección de anomalías
        features = self.pipeline.named_steps['features'].transform(X)
        anomaly_scores = self.iso_forest.decision_function(features)
        threshold_score = np.percentile(anomaly_scores, anomaly_percentile)
        
        results = {
            "request_text": X["request_text"] if isinstance(X, pd.DataFrame) else X,
            "predicted_type": self.label_encoder.inverse_transform(y_pred),
            "confidence": max_probs,
            "is_uncertain": uncertain_mask,
            "anomaly_score": anomaly_scores,
            "is_outlier": anomaly_scores < threshold_score
        }
        
        # Añadir información adicional si hay etiquetas originales
        if isinstance(X, pd.DataFrame) and "type" in X.columns:
            results.update({
                "original_type": X["type"],
                "discrepancy": (X["type"] != results["predicted_type"]),
                "suggested_type": results["predicted_type"]  
            })
            
            # Marcador combinado de problemas
            results["needs_review"] = (
                results["is_uncertain"] | 
                results["is_outlier"] | 
                results["discrepancy"]
            )
        
        return pd.DataFrame(results)
        
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
