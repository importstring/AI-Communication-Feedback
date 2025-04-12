import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from rstparser import RSTParser

nlp = spacy.load("en_core_web_trf")

class StructureAnalyzer:
    def analyze_structure(self, text):
        doc = nlp(text)
        coherence_score = self._calculate_coherence(doc)
        transition_density = len([sent for sent in doc.sents if self._is_transition(sent)])/len(list(doc.sents))
        
        return {
            'coherence_index': coherence_score,
            'transition_score': transition_density,
            'hierarchy_depth': self._measure_hierarchy(doc)
        }

    def _calculate_coherence(self, doc):
        """Entity grid coherence metric (Barzilay & Lapata, 2008)"""
        entity_grid = defaultdict(list)
        for sent_idx, sent in enumerate(doc.sents):
            for token in sent:
                if token.ent_type_:
                    role = 'S' if token.dep_ == 'nsubj' else 'O' if token.dep_ == 'dobj' else 'X'
                    entity_grid[token.text].append((sent_idx, role))
        
        transitions = []
        for entity, mentions in entity_grid.items():
            for i in range(1, len(mentions)):
                prev_role = mentions[i-1][1]
                curr_role = mentions[i][1]
                transitions.append(f"{prev_role}→{curr_role}")
        
        valid_transitions = [t for t in transitions if t in ['S→S', 'S→O', 'O→S']]
        return len(valid_transitions) / len(transitions) if transitions else 0

    def _measure_hierarchy(self, doc):
        """Rhetorical Structure Theory depth analysis"""
        parser = RSTParser()
        tree = parser.parse(doc.text)
        return tree.max_depth()
