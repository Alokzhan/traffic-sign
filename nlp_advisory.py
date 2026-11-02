"""Natural Language Processing (NLP) Advisory & Semantic Search Engine for Traffic Signs.
Provides semantic hierarchies, contextual driver advisories (English + Hindi/Hinglish),
legal rules, severity levels, and TF-IDF natural language query search.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from common import CLASS_NAMES, OTHER_CLASS_NAME, NUM_CLASSES

# Complete Knowledge Base for all 44 classes
SIGN_KNOWLEDGE_BASE = {
    0: {
        "name": "Speed limit 20",
        "category": "Speed Regulation",
        "severity": "HIGH",
        "action_en": "SPEED REDUCTION: Maximum allowable speed is 20 km/h. High pedestrian/calmed traffic zone. Prepare for immediate braking.",
        "action_hi": "Speed 20 km/h ke andar rakhein. Paidal chalne walo ka dhyan rakhein, emergency brake ke liye taiyar rahein.",
        "keywords": "speed limit 20 slow traffic residential zone pedestrian living street 20kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-20)"
    },
    1: {
        "name": "Speed limit 30",
        "category": "Speed Regulation",
        "severity": "HIGH",
        "action_en": "SPEED REDUCTION: Maximum allowable speed is 30 km/h. Typical residential, school, or hospital zone.",
        "action_hi": "Gadi ki speed 30 km/h se kam rakhein. School ya residential zone ho sakta hai.",
        "keywords": "speed limit 30 residential school zone hospital 30kmh slow",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-30)"
    },
    2: {
        "name": "Speed limit 50",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "URBAN SPEED LIMIT: Standard city/urban maximum speed is 50 km/h. Watch for cross-traffic and traffic lights.",
        "action_hi": "City/Urban limit: Speed 50 km/h tak rakhein. Crossing aur traffic signals par dhyan dein.",
        "keywords": "speed limit 50 urban city standard speed 50kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-50)"
    },
    3: {
        "name": "Speed limit 60",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "SPEED LIMIT 60: Maximum allowable speed is 60 km/h on urban arterial roads or construction bypasses.",
        "action_hi": "Speed 60 km/h ke andar rakhein. Arterial road ya bypass zone.",
        "keywords": "speed limit 60 arterial bypass 60kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-60)"
    },
    4: {
        "name": "Speed limit 70",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "SPEED LIMIT 70: Maximum allowable speed is 70 km/h on semi-urban or rural transition roads.",
        "action_hi": "Speed 70 km/h se zyada na karein. Highway transition road.",
        "keywords": "speed limit 70 rural transition highway approach 70kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-70)"
    },
    5: {
        "name": "Speed limit 80",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "SPEED LIMIT 80: Maximum allowable speed is 80 km/h. Standard highway construction or rural route speed.",
        "action_hi": "Speed 80 km/h tak seemit rakhein. Highway zone.",
        "keywords": "speed limit 80 highway rural route 80kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-80)"
    },
    6: {
        "name": "End of speed limit 80",
        "category": "De-restriction",
        "severity": "INFORMATIONAL",
        "action_en": "LIMIT ENDS: 80 km/h speed restriction has ended. Revert to standard default route speed limits.",
        "action_hi": "80 km/h ki limit khatam. Road ki standard default speed par chal sakte hain.",
        "keywords": "end of speed limit 80 restriction lifted resume default speed",
        "legal_rule": "StVO § 41 (Sign 278-80)"
    },
    7: {
        "name": "Speed limit 100",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "HIGHWAY SPEED LIMIT: Maximum speed 100 km/h. Standard rural non-urban two-lane road limit.",
        "action_hi": "Highway limit: Speed 100 km/h tak rakhein. Lane discipline follow karein.",
        "keywords": "speed limit 100 highway rural express 100kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-100)"
    },
    8: {
        "name": "Speed limit 120",
        "category": "Speed Regulation",
        "severity": "MODERATE",
        "action_en": "EXPRESSWAY SPEED LIMIT: Maximum speed 120 km/h on dual carriageway or Autobahn section.",
        "action_hi": "Expressway limit: Speed 120 km/h ke andar rakhein.",
        "keywords": "speed limit 120 motorway expressway autobahn 120kmh",
        "legal_rule": "StVO § 41 Abs. 1 (Sign 274-120)"
    },
    9: {
        "name": "No passing",
        "category": "Prohibitory",
        "severity": "HIGH",
        "action_en": "OVERTAKING FORBIDDEN: Do not overtake or pass any multi-track motorized vehicles. Maintain lane position.",
        "action_hi": "Overtake karna mana hai. Aage wali gadi ko cross na karein, apni lane mein rahein.",
        "keywords": "no passing no overtaking forbid pass cars vehicles double line",
        "legal_rule": "StVO § 41 (Sign 276)"
    },
    10: {
        "name": "No passing for vehicles > 3.5t",
        "category": "Prohibitory",
        "severity": "HIGH",
        "action_en": "HEAVY VEHICLE PASSING PROHIBITED: Heavy commercial vehicles (> 3.5 tons) cannot overtake. Light passenger cars unaffected.",
        "action_hi": "Bhaari gaadiyo (> 3.5 ton) ke liye overtake karna mana hai.",
        "keywords": "no passing trucks heavy goods vehicles 3.5t prohibited",
        "legal_rule": "StVO § 41 (Sign 277)"
    },
    11: {
        "name": "Right-of-way at intersection",
        "category": "Priority",
        "severity": "HIGH",
        "action_en": "INTERSECTION PRIORITY: You have right-of-way at the upcoming single intersection only. Maintain cautious speed.",
        "action_hi": "Agli crossing par aapko pehla rasta (priority) milega. Cautious rahein.",
        "keywords": "right of way intersection priority single junction priority",
        "legal_rule": "StVO § 42 (Sign 301)"
    },
    12: {
        "name": "Priority road",
        "category": "Priority",
        "severity": "MODERATE",
        "action_en": "PRIORITY ROAD CONTINUES: You have ongoing right-of-way along this primary roadway until an end-of-priority sign.",
        "action_hi": "Priority road chalu hai: Aage aane wale sabhi junctions par aapka haq pehle hai.",
        "keywords": "priority road yellow diamond ongoing right of way main road",
        "legal_rule": "StVO § 42 (Sign 306)"
    },
    13: {
        "name": "Yield",
        "category": "Priority",
        "severity": "CRITICAL",
        "action_en": "GIVE WAY: Decelerate smoothly and yield right-of-way to intersecting traffic. Stop if vehicles are approaching.",
        "action_hi": "Rasta dein (Yield): Speed kam karein, cross road ki gaadiyo ko nikalne dein, zaroorat padne par rokein.",
        "keywords": "yield give way inverted triangle priority crossing stop if busy",
        "legal_rule": "StVO § 41 (Sign 205)"
    },
    14: {
        "name": "Stop",
        "category": "Priority / Mandatory",
        "severity": "CRITICAL",
        "action_en": "MANDATORY FULL STOP: Bring the vehicle to a complete 0 km/h stop before the stop line. Yield to all oncoming traffic.",
        "action_hi": "Purna Viraam (Stop): Gaadi poori tarah rokein (0 km/h). Dono taraf dekh kar raasta clear hone par hi chalein.",
        "keywords": "stop octagonal red full stop halt mandatory zero speed",
        "legal_rule": "StVO § 41 (Sign 206)"
    },
    15: {
        "name": "No vehicles",
        "category": "Prohibitory",
        "severity": "CRITICAL",
        "action_en": "ALL VEHICLES PROHIBITED: Closed to all vehicular traffic in both directions (pedestrians only).",
        "action_hi": "Sabhi gaadiyo ka pravesh nishedh: Yahan koi bhi gadi nahi le ja sakte.",
        "keywords": "no vehicles red circle white pedestrian zone closed both directions",
        "legal_rule": "StVO § 41 (Sign 250)"
    },
    16: {
        "name": "Vehicles > 3.5t prohibited",
        "category": "Prohibitory",
        "severity": "HIGH",
        "action_en": "HEAVY VEHICLES BANNED: Vehicles with gross vehicle weight exceeding 3.5 tons cannot enter.",
        "action_hi": "3.5 ton se bhari gaadiyo ka aana mana hai.",
        "keywords": "heavy goods trucks 3.5t tonnage weight ban prohibited",
        "legal_rule": "StVO § 41 (Sign 253)"
    },
    17: {
        "name": "No entry",
        "category": "Prohibitory",
        "severity": "CRITICAL",
        "action_en": "DO NOT ENTER: Wrong-way hazard. Entry strictly prohibited in this direction (e.g. one-way street exit).",
        "action_hi": "Pravesh Nishedh (No Entry): Galat disha (wrong way) danger. Is raste par aage na jayein.",
        "keywords": "no entry do not enter wrong way one way street red bar horizontal",
        "legal_rule": "StVO § 41 (Sign 267)"
    },
    18: {
        "name": "General caution",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "HAZARD AHEAD: Unspecified roadway danger ahead. Reduce speed, scan the roadway, and increase following distance.",
        "action_hi": "Savdhani (Caution): Aage anjaan khatra ho sakta hai. Speed kam karein aur dhyan se dekhein.",
        "keywords": "general caution danger warning exclamation mark hazard road",
        "legal_rule": "StVO § 40 (Sign 101)"
    },
    19: {
        "name": "Dangerous curve left",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "SHARP LEFT BEND: Sharp left curve approaching. Decelerate before steering into the corner to avoid lateral skid.",
        "action_hi": "Aage baayein (left) taraf tez mod hai. Modne se pehle speed kam karein.",
        "keywords": "dangerous curve left sharp bend turn corner skid hazard",
        "legal_rule": "StVO § 40 (Sign 103-10)"
    },
    20: {
        "name": "Dangerous curve right",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "SHARP RIGHT BEND: Sharp right curve approaching. Decelerate before the apex to maintain optimal tire grip.",
        "action_hi": "Aage daayein (right) taraf tez mod hai. Speed kam karke modhein.",
        "keywords": "dangerous curve right sharp bend turn corner hazard",
        "legal_rule": "StVO § 40 (Sign 103-20)"
    },
    21: {
        "name": "Double curve",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "WINDING ROAD: S-curve sequence ahead. Maintain moderate speed through the multiple alternating bends.",
        "action_hi": "Aage dohra mod (S-curve) hai. Gaadi control mein rakhein.",
        "keywords": "double curve s bend winding road chicane alternating curves",
        "legal_rule": "StVO § 40 (Sign 105-10)"
    },
    22: {
        "name": "Bumpy road",
        "category": "Warning / Hazard",
        "severity": "MODERATE",
        "action_en": "UNEVEN SURFACE: Potholes or surface roughness ahead. Slow down to avoid suspension damage and loss of control.",
        "action_hi": "Khabad-khabad (bumpy) rasta: Gadi dheere karein taaki suspension aur santulan bana rahe.",
        "keywords": "bumpy road potholes uneven surface road damage dips",
        "legal_rule": "StVO § 40 (Sign 112)"
    },
    23: {
        "name": "Slippery road",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "LOW TRACTION: Road surface is slick (rain, mud, oil). Avoid abrupt braking, aggressive steering, or rapid acceleration.",
        "action_hi": "Phisalan bhara rasta: Achanak brake ya tez steering na karein, gadi phisal sakti hai.",
        "keywords": "slippery road skid traction wet slick rain oil aquaplaning",
        "legal_rule": "StVO § 40 (Sign 114)"
    },
    24: {
        "name": "Road narrows on right",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "LANE MERGE: Right lane narrows or terminates ahead. Signal and execute a smooth zipper merge to the left.",
        "action_hi": "Daayein taraf se road patli ho rahi hai: Left lane mein merge hone ki taiyari karein.",
        "keywords": "road narrows right lane merge bottleneck zipper",
        "legal_rule": "StVO § 40 (Sign 120)"
    },
    25: {
        "name": "Road work",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "CONSTRUCTION ZONE: Active workers and road machinery ahead. Reduce speed and obey temporary detour markings.",
        "action_hi": "Aage kaam chal raha hai (Road Work): Speed kam karein, workers aur machines ka dhyan rakhein.",
        "keywords": "road work construction workers machinery detour hazard maintenance",
        "legal_rule": "StVO § 40 (Sign 123)"
    },
    26: {
        "name": "Traffic signals",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "TRAFFIC LIGHTS AHEAD: Traffic light signals control the upcoming intersection. Prepare to stop on amber or red.",
        "action_hi": "Aage traffic signals (battiyaan) hain: Red light par rukne ke liye taiyar rahein.",
        "keywords": "traffic signals traffic lights red yellow green signal stop intersection",
        "legal_rule": "StVO § 40 (Sign 131)"
    },
    27: {
        "name": "Pedestrians",
        "category": "Warning / Hazard",
        "severity": "CRITICAL",
        "action_en": "PEDESTRIAN RISK: High pedestrian crossing activity. Scan curbsides and be ready to yield immediately.",
        "action_hi": "Paidal yatri (Pedestrians): Raste par log cross kar sakte hain, speed kam rakhein aur rasta dein.",
        "keywords": "pedestrians crossing walking zebra people crosswalk hazard",
        "legal_rule": "StVO § 40 (Sign 133)"
    },
    28: {
        "name": "Children crossing",
        "category": "Warning / Hazard",
        "severity": "CRITICAL",
        "action_en": "SCHOOL ZONE: Children near school, playground, or transit stop. Drive with extreme caution and low speed.",
        "action_hi": "Bachhe cross kar sakte hain: School/playground area. Bahut savdhani aur kam speed se chalayein.",
        "keywords": "children crossing school playground kids students watch out",
        "legal_rule": "StVO § 40 (Sign 136)"
    },
    29: {
        "name": "Bicycles crossing",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "CYCLIST CROSSING: Cyclists crossing or entering traffic. Check blind spots and grant clearance.",
        "action_hi": "Cycle chalane wale: Cycle lane ya crossing ho sakti hai, blind spots check karein.",
        "keywords": "bicycles crossing cyclists bike lane bicycle path",
        "legal_rule": "StVO § 40 (Sign 138)"
    },
    30: {
        "name": "Beware of ice/snow",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "WINTER CONDITIONS: Risk of black ice or hard-packed snow. Increase safe stopping distance threefold.",
        "action_hi": "Baraf/Ice ka khatra: Phisalan bahut zyada ho sakti hai, braking distance 3 guna badha lein.",
        "keywords": "beware ice snow snowflake black ice frost frozen winter slippery",
        "legal_rule": "StVO § 40 (Sign 144)"
    },
    31: {
        "name": "Wild animals crossing",
        "category": "Warning / Hazard",
        "severity": "HIGH",
        "action_en": "DEER/ANIMAL CROSSING: Forest corridor with frequent animal crossings, especially during dusk and dawn.",
        "action_hi": "Jungli janwar cross kar sakte hain: Jungle corridor, shaam aur subah khas dhyan rakhein.",
        "keywords": "wild animals crossing deer forest wildlife fauna crossing",
        "legal_rule": "StVO § 40 (Sign 142)"
    },
    32: {
        "name": "End of all speed and passing limits",
        "category": "De-restriction",
        "severity": "INFORMATIONAL",
        "action_en": "RESTRICTIONS CLEARED: All previous temporary speed and passing bans are lifted. Default road codes apply.",
        "action_hi": "Sabhi pabandiyan samapt: Pehle ke speed aur overtaking restrictions ab khatam.",
        "keywords": "end of all speed and passing limits de restriction clear autobahn",
        "legal_rule": "StVO § 41 (Sign 282)"
    },
    33: {
        "name": "Turn right ahead",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "MANDATORY RIGHT TURN: You must execute a right turn at the upcoming intersection. Do not go straight or left.",
        "action_hi": "Aage daayein (right) modna anivarya hai: Seedhe ya left na jayein.",
        "keywords": "turn right ahead mandatory arrow blue circle right turn only",
        "legal_rule": "StVO § 41 (Sign 209-20)"
    },
    34: {
        "name": "Turn left ahead",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "MANDATORY LEFT TURN: You must execute a left turn at the upcoming intersection. Yield to oncoming straight traffic.",
        "action_hi": "Aage baayein (left) modna anivarya hai: Samne se aane wali gaadiyo ko dekh kar mudein.",
        "keywords": "turn left ahead mandatory arrow blue circle left turn only",
        "legal_rule": "StVO § 41 (Sign 209-10)"
    },
    35: {
        "name": "Ahead only",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "STRAIGHT AHEAD ONLY: Mandatory straight travel. No turns permitted at this junction.",
        "action_hi": "Sirf seedhe jayein (Ahead only): Yahan se left ya right modna mana hai.",
        "keywords": "ahead only straight forward no turn permitted arrow up",
        "legal_rule": "StVO § 41 (Sign 209-30)"
    },
    36: {
        "name": "Go straight or right",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "STRAIGHT OR RIGHT: Permitted paths are straight or right turn. Left turns strictly prohibited.",
        "action_hi": "Seedhe ya daayein jayein: Left turn lena mana hai.",
        "keywords": "go straight or right straight and right mandatory options",
        "legal_rule": "StVO § 41 (Sign 214-20)"
    },
    37: {
        "name": "Go straight or left",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "STRAIGHT OR LEFT: Permitted paths are straight or left turn. Right turns strictly prohibited.",
        "action_hi": "Seedhe ya baayein jayein: Right turn lena mana hai.",
        "keywords": "go straight or left straight and left mandatory options",
        "legal_rule": "StVO § 41 (Sign 214-10)"
    },
    38: {
        "name": "Keep right",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "KEEP RIGHT OF OBSTACLE: Pass to the right of the traffic island, divider, or upcoming road barrier.",
        "action_hi": "Daayein taraf se niklein (Keep Right): Divider ya island ke right side se jayein.",
        "keywords": "keep right traffic island pass right divider arrow downward right",
        "legal_rule": "StVO § 41 (Sign 222-20)"
    },
    39: {
        "name": "Keep left",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "KEEP LEFT OF OBSTACLE: Pass to the left of the traffic island, barrier, or construction splitter.",
        "action_hi": "Baayein taraf se niklein (Keep Left): Divider ke left side se jayein.",
        "keywords": "keep left traffic island pass left divider arrow downward left",
        "legal_rule": "StVO § 41 (Sign 222-10)"
    },
    40: {
        "name": "Roundabout mandatory",
        "category": "Mandatory",
        "severity": "HIGH",
        "action_en": "ROUNDABOUT MANDATORY: Counter-clockwise rotary traffic. Yield to vehicles already inside the circle. Indicate right when exiting.",
        "action_hi": "Gol chakkar (Roundabout): Circle ke andar chal rahi gaadiyo ko rasta dein. Exit lene se pehle indicator dein.",
        "keywords": "roundabout rotary traffic circle circular arrows mandatory yield inside",
        "legal_rule": "StVO § 41 (Sign 215)"
    },
    41: {
        "name": "End of no passing",
        "category": "De-restriction",
        "severity": "INFORMATIONAL",
        "action_en": "PASSING PERMITTED AGAIN: The overtaking prohibition for cars has ended. Pass only when clear and legal.",
        "action_hi": "Overtake karne ki pabandi khatam: Raasta saaf hone par aage nikal sakte hain.",
        "keywords": "end of no passing overtaking allowed again resume passing",
        "legal_rule": "StVO § 41 (Sign 280)"
    },
    42: {
        "name": "End of no passing for vehicles > 3.5t",
        "category": "De-restriction",
        "severity": "INFORMATIONAL",
        "action_en": "HEAVY VEHICLE PASSING ALLOWED: Overtaking ban for heavy commercial trucks is officially lifted.",
        "action_hi": "Bhaari gaadiyo (> 3.5t) ke overtake karne ki pabandi samapt.",
        "keywords": "end of no passing trucks heavy vehicles lifted",
        "legal_rule": "StVO § 41 (Sign 281)"
    },
    43: {
        "name": "Other / not a sign",
        "category": "Out-of-Distribution / Non-Sign",
        "severity": "INFORMATIONAL",
        "action_en": "OBJECT NOT A SUPPORTED SIGN: The camera detected an element (such as a traffic light, lamppost, billboard, or non-sign object). Do not alter vehicle heading or velocity based on traffic-sign rules.",
        "action_hi": "Yeh koi supported traffic sign nahi hai: Camera ne traffic light, khamba, ya koi anya vastu dekhi hai. Gadi ki speed ya direction sign samajh kar na badlein.",
        "keywords": "other not a sign traffic light lamp post background noise distractor billboard object",
        "legal_rule": "Open-Set Safety Rejection Filter"
    }
}


# Pre-build TF-IDF semantic search model across all sign descriptions
_corpus = []
for cid in range(NUM_CLASSES):
    info = SIGN_KNOWLEDGE_BASE.get(cid, {})
    doc = f"{info.get('name', '')} {info.get('category', '')} {info.get('action_en', '')} {info.get('keywords', '')}"
    _corpus.append(doc)

_vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", lowercase=True)
_tfidf_matrix = _vectorizer.fit_transform(_corpus)


def get_sign_advisory(class_idx: int) -> dict:
    """Retrieve complete NLP semantic advisory for a given predicted class index."""
    if class_idx in SIGN_KNOWLEDGE_BASE:
        return SIGN_KNOWLEDGE_BASE[class_idx]
    return {
        "name": CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown",
        "category": "General",
        "severity": "MODERATE",
        "action_en": "Exercise defensive driving caution.",
        "action_hi": "Savdhani se gaadi chalayein.",
        "legal_rule": "General Safety"
    }


def nlp_search(query: str, top_k: int = 3) -> list:
    """Natural Language Query Semantic Search:
    User types free-form text ('school kids', 'traffic light', '30 speed', 'no overtake')
    and NLP vector similarity finds the most relevant traffic signs with rules.
    """
    if not query or not query.strip():
        return []
    
    clean_q = re.sub(r"[^\w\s]", " ", query.lower()).strip()
    q_vec = _vectorizer.transform([clean_q])
    sims = cosine_similarity(q_vec, _tfidf_matrix)[0]
    
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        score = float(sims[idx])
        if score > 0.04:  # Relevance threshold
            info = dict(SIGN_KNOWLEDGE_BASE[idx])
            info["class_idx"] = int(idx)
            info["relevance_score"] = round(score, 3)
            results.append(info)
    return results


def generate_driver_report(class_idx: int, conf: float, entropy: float, is_uncertain: bool) -> dict:
    """Generate a synthesized autonomous vehicle narrative summary report."""
    info = get_sign_advisory(class_idx)
    name = info.get("name", "Unknown")
    
    if class_idx == 43 or "Other" in name:
        narrative = (f"[Safety Rejection] Detected out-of-distribution object with {conf:.1%} confidence. "
                     f"Verdict: Not an actionable road sign. Vehicle motion remains unaffected.")
    elif is_uncertain:
        narrative = (f"[Uncertainty Warning] Best guess is '{name}' with {conf:.1%} confidence "
                     f"(Entropy: {entropy:.2f}). Below safety threshold. Driver caution advised.")
    else:
        narrative = (f"[Perception Confirmed] Detected '{name}' with {conf:.1%} confidence "
                     f"(Entropy: {entropy:.2f}). Safety Rule: {info.get('action_en')}")
    
    return {
        "narrative": narrative,
        "advisory": info
    }
