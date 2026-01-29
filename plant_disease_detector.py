import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import warnings
warnings.filterwarnings('ignore')

# Configuration
dataset_path = r"C:\Users\manis\OneDrive\Desktop\plant-disease\plantvillage_dataset\color"
img_size = (240, 240)  
epochs = 5
batch_size = 8

# Disease solutions database
disease_solutions = {
    # Apple
    "Apple___Apple_scab": {
        "symptoms": "Dark, scabby lesions on leaves and fruit. Lesions are circular or irregular, dark green initially, then turning brown or black with a velvety appearance. Affected leaves may show yellow halos and eventually drop. Fruit becomes misshapen with corky brown scabs.",
        "causes": "Fungal infection caused by Venturia inaequalis. This fungus overwinters in fallen leaves and infected fruit. Spores are released during wet spring weather and spread by wind and rain. The disease thrives in cool, humid conditions (16-24°C) with frequent rainfall. Poor air circulation and overhead watering create ideal conditions for infection. The fungus penetrates leaf tissue through stomata and colonizes, producing more spores that spread the disease.",
        "solutions": [
            "Apply fungicides containing myclobutanil or captan at the first sign of infection",
            "Remove and destroy all fallen leaves and infected fruit immediately",
            "Improve air circulation by proper pruning and spacing of trees",
            "Use resistant apple varieties like 'Liberty', 'Freedom', or 'Enterprise'",
            "Apply sulfur-based fungicides in early spring before bud break",
            "Avoid overhead watering to reduce leaf wetness",
            "Apply lime sulfur during dormant season to kill overwintering spores"
        ],
        "prevention": "Plant disease-resistant varieties, maintain proper tree spacing (15-20 feet apart), prune regularly to improve air circulation, clean up fallen leaves and debris in fall, avoid overhead irrigation, apply preventive fungicide sprays during wet weather periods, monitor weather forecasts and apply fungicides before rain events, maintain tree health with proper fertilization and watering to reduce stress susceptibility"
    },
    "Apple___Black_rot": {
        "symptoms": "Brown to black circular lesions on fruit and leaves. Lesions typically have concentric rings and a sunken center. On fruit, rot can spread rapidly. Leaves show brown to black spots with darker borders. Cankers may appear on twigs and branches.",
        "causes": "Fungal infection caused by Botryosphaeria obtusa. The fungus survives in dead wood, bark, and infected fruit mummies. Spores are produced in moist conditions and spread by rain splash and wind. Disease development is favored by warm, humid weather (24-29°C) and extended wetness periods. Wounds from pruning, insects, or hail provide entry points. The fungus can survive in cankers for years, producing spores annually.",
        "solutions": [
            "Remove and destroy all infected fruit, leaves, and cankered wood immediately",
            "Apply copper-based fungicides or captan during bloom and petal fall",
            "Prune out cankered branches 6-8 inches below visible symptoms",
            "Apply thiophanate-methyl fungicides for severe infections",
            "Use sterile pruning tools and disinfect between cuts",
            "Avoid wounding trees during wet weather",
            "Apply fungicides preventively during wet spring periods"
        ],
        "prevention": "Prune trees during dry weather to minimize wound infections, maintain tree health with proper fertilization to reduce stress susceptibility, remove and destroy mummified fruit and dead wood, apply preventive fungicide sprays during wet weather periods, avoid overhead irrigation, plant resistant varieties when available, space trees properly for air circulation, monitor for cankers and remove them promptly, disinfect pruning tools between trees, apply dormant sprays to kill overwintering spores"
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": "Yellow-orange spots develop on leaves and fruit, often with a reddish border. Small tubular or horn-like structures (galls) appear on infected tissue. Fruit may become distorted. Cedar trees show brownish gelatinous masses on branches.",
        "causes": "Fungal infection caused by Gymnosporangium juniperi-virginianae. This rust fungus requires two hosts: apple/juniper and eastern red cedar. The fungus overwinters on cedar trees as galls. In spring, spores are released and infect apple leaves during wet weather. Apple infections produce spores that reinfect cedar trees. The disease cycle requires both hosts and cannot complete without them. Cool, wet spring weather (13-21°C) favors infection.",
        "solutions": [
            "Remove nearby eastern red cedar trees within 1-2 miles if possible",
            "Apply fungicides containing myclobutanil or triadimefon during early spring",
            "Use resistant apple varieties like 'Enterprise', 'Goldrush', or 'Liberty'",
            "Apply sulfur fungicides during wet spring periods",
            "Prune infected branches and destroy fallen leaves",
            "Apply preventive sprays before rain events",
            "Use copper fungicides as organic alternative"
        ],
        "prevention": "Plant rust-resistant apple varieties, remove or relocate eastern red cedar trees away from apple orchards, apply preventive fungicide sprays during wet spring weather, monitor weather forecasts and time applications before rain, maintain proper tree spacing for air circulation, clean up fallen leaves in fall, prune trees to improve air flow and reduce humidity, use reflective mulches to reduce spore germination, plant apple trees away from cedar windbreaks"
    },
    "Apple___healthy": {
        "symptoms": "Leaves are uniformly green with no spots or discoloration. Fruit is firm, unblemished, and properly colored for the variety. No wilting, lesions, or abnormal growth patterns visible.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering schedule",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning schedule"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Blueberry
    "Blueberry___healthy": {
        "symptoms": "Leaves show deep green color with smooth edges. Berries are uniformly colored and firm. No visible spots, wilting, or abnormal growth patterns.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Cherry
    "Cherry_(including_sour)___healthy": {
        "symptoms": "Leaves are green and smooth with no spots or lesions. Fruit is firm and properly colored. Branches show no cankers or abnormal growth.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White powdery coating appears on leaves, stems, and developing fruit. Infected leaves curl and become distorted. Fruit may be covered with white powder or have a grayish appearance. Affected areas may turn reddish or purplish.",
        "causes": "Fungal infection caused by Podosphaera clandestina. The fungus survives as mycelium in buds and on bark. Spores are produced in warm, dry conditions and spread by wind. Disease development is favored by warm days (21-27°C) and cool nights with high humidity. Dense foliage, poor air circulation, and nitrogen deficiency increase susceptibility. The fungus grows on leaf surfaces without penetrating tissue, feeding on plant nutrients.",
        "solutions": [
            "Apply sulfur-based fungicides or potassium bicarbonate sprays",
            "Remove infected plant material to reduce spore sources",
            "Improve air circulation through proper pruning",
            "Avoid overhead watering to keep foliage dry",
            "Apply fungicides containing myclobutanil for severe infections",
            "Use neem oil or horticultural oils as organic alternatives",
            "Apply treatments in early morning or late evening to avoid leaf burn"
        ],
        "prevention": "Plant resistant cherry varieties, maintain proper tree spacing for air circulation, prune regularly to improve light penetration and reduce humidity, avoid excessive nitrogen fertilization, water at soil level to keep leaves dry, apply preventive sulfur sprays during warm, humid weather, monitor for early symptoms and treat promptly, maintain tree health with balanced fertilization, remove and destroy infected leaves in fall, use reflective mulches to reduce fungal growth"
    },
    # Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": "Gray to tan rectangular lesions develop on leaves, aligned with leaf veins. Lesions have distinct dark borders and may coalesce into larger areas. Infected leaves become tattered and may dry up. Symptoms start on lower leaves and progress upward.",
        "causes": "Fungal infection caused by Cercospora zeae-maydis. The fungus survives on crop residue and infected seeds. Spores are spread by wind, rain, and irrigation water. Disease development is favored by warm temperatures (24-29°C), high humidity, and prolonged leaf wetness. Dense planting, poor air circulation, and nitrogen deficiency increase susceptibility. The fungus penetrates through stomata and colonizes leaf tissue, producing conidia that spread to healthy plants.",
        "solutions": [
            "Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole at first sign of symptoms",
            "Use resistant corn hybrids specifically bred for gray leaf spot resistance",
            "Practice crop rotation with non-host crops for at least 2 years",
            "Remove and destroy crop debris after harvest by plowing under or burning",
            "Improve field drainage to reduce prolonged leaf wetness",
            "Apply fungicides preventively during tasseling and silking stages",
            "Use balanced fertilization to avoid nitrogen deficiency stress"
        ],
        "prevention": "Plant resistant varieties, rotate crops with non-grass crops for 2+ years, till under crop residue after harvest, avoid overhead irrigation, maintain proper plant spacing for air circulation, apply balanced nitrogen fertilization, monitor fields regularly during growing season, apply preventive fungicide sprays during humid weather periods, plant early to avoid peak disease pressure, use certified disease-free seed"
    },
    "Corn_(maize)___Common_rust_": {
        "symptoms": "Orange to rusty-brown pustules appear on both leaf surfaces, initially appearing as small bumps. Leaves develop a rust-colored powder (spores) when pustules break open. Severe infection causes leaves to dry up prematurely.",
        "causes": "Fungal infection caused by Puccinia sorghi. The fungus requires two hosts: corn and oxalis (wood sorrel). Spores overwinter on oxalis plants and infect corn in spring. Disease spreads rapidly through wind-borne spores. Warm temperatures (21-29°C) with high humidity and dew favor development. The fungus penetrates leaf tissue and forms pustules containing thousands of spores. Late-planted corn is more susceptible due to smaller plants.",
        "solutions": [
            "Apply fungicides containing propiconazole, tebuconazole, or azoxystrobin",
            "Use resistant corn hybrids with rust resistance genes",
            "Remove oxalis weeds from fields and borders",
            "Apply fungicides preventively during tasseling stage",
            "Use copper fungicides for organic control",
            "Apply treatments when first pustules appear",
            "Increase fungicide frequency during humid weather"
        ],
        "prevention": "Plant rust-resistant corn hybrids, control oxalis weeds around fields, avoid late planting to reduce susceptibility, apply preventive fungicide sprays during humid weather periods, monitor fields regularly for early symptoms, use balanced fertilization to maintain plant health, practice crop rotation to break disease cycle, plant early maturing varieties to avoid peak disease pressure, maintain proper plant spacing for air circulation, apply fungicides at tasseling if weather favors disease"
    },
    "Corn_(maize)___healthy": {
        "symptoms": "Leaves are vibrant green with clean, uniform appearance. No spots, lesions, or pustules visible. Plant shows vigorous growth with no discoloration.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper field management"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long, tan elliptical lesions appear with dark borders, typically 1-6 inches long. Lesions may have a grayish center. Affected leaves become shredded as disease progresses. Multiple lesions cause leaves to wither and die.",
        "causes": "Fungal infection caused by Exserohilum turcicum (formerly Helminthosporium turcicum). The fungus survives on infected corn residue and seeds. Spores are spread by wind and rain splash. Disease thrives in warm, humid conditions (24-29°C) with extended leaf wetness. Dense planting and nitrogen deficiency increase susceptibility. The fungus produces conidia that spread rapidly within fields. Resistant varieties have been developed with Ht genes.",
        "solutions": [
            "Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole",
            "Use resistant corn hybrids with Ht genes for northern leaf blight",
            "Practice crop rotation with non-host crops",
            "Remove and destroy crop residue after harvest",
            "Apply fungicides at tasseling and silking stages",
            "Use balanced fertilization to avoid nitrogen stress",
            "Apply preventive sprays during wet weather periods"
        ],
        "prevention": "Plant resistant hybrids with multiple Ht genes, rotate crops with non-grass crops for 2+ years, till under crop residue after harvest, avoid dense planting that reduces air circulation, apply balanced nitrogen fertilization, monitor weather and apply preventive fungicides before wet periods, plant early to mature before peak disease pressure, use certified disease-free seed, scout fields regularly for early symptoms, maintain proper field drainage to reduce leaf wetness"
    },
    # Grape
    "Grape___Black_rot": {
        "symptoms": "Dark brown to black spots appear on leaves initially with circular shape. Lesions expand rapidly and may develop concentric rings. Fruit turns brown to black and becomes mummified. Leaves often have a yellow halo around lesions.",
        "causes": "Fungal infection caused by Guignardia bidwellii. The fungus survives in mummified berries and infected canes. Spores are released during wet spring weather and spread by rain splash. Disease development requires warm, humid conditions (24-29°C) with extended wetness periods (12+ hours). The fungus penetrates through stomata and wounds. Infected fruit mummies serve as primary inoculum for the next season.",
        "solutions": [
            "Apply fungicides containing myclobutanil, captan, or mancozeb",
            "Remove and destroy all mummified fruit and infected leaves",
            "Prune infected canes during dormant season",
            "Apply fungicides at 10-14 day intervals during wet weather",
            "Use copper fungicides for organic control",
            "Improve air circulation through proper pruning",
            "Apply preventive sprays before rain events"
        ],
        "prevention": "Remove and destroy mummified berries and infected canes in winter, apply dormant sprays to kill overwintering spores, use resistant grape varieties when available, maintain proper vine spacing for air circulation, prune vines to improve light penetration and reduce humidity, apply preventive fungicide sprays during wet spring weather, monitor weather forecasts and time applications before rain, avoid overhead irrigation, use reflective mulches to reduce spore germination, maintain vine health with balanced fertilization"
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Leaves show tiger-striped or sectored appearance with alternating black and white/gray patches. Fruit develops black streaks and spots. Affected berries shrivel. Older wood may show internal discoloration when cut.",
        "causes": "Fungal infection caused by a complex of fungi including Phaeomoniella chlamydospora and Phaeoacremonium aleophilum. The fungi enter through pruning wounds and grow slowly in vascular tissue. Disease symptoms appear years after initial infection. Warm, humid conditions favor fungal growth and symptom development. The fungi block water and nutrient transport, causing leaf discoloration and fruit symptoms. Infected vines decline over several years.",
        "solutions": [
            "Prune during dry weather to minimize wound infections",
            "Apply wound protectants like Bordeaux mixture after pruning",
            "Remove and destroy severely infected vines",
            "Improve soil drainage to reduce stress",
            "Apply fungicides containing tebuconazole systemically",
            "Use hot water treatment for propagation material",
            "Avoid pruning during wet weather"
        ],
        "prevention": "Prune vines during dry weather only, apply wound sealants immediately after pruning, use certified disease-free planting material, avoid over-fertilization that promotes excessive growth, maintain proper soil drainage, monitor vines for early symptoms and remove infected plants, disinfect pruning tools between vines, plant resistant rootstocks when available, avoid wounding vines during wet periods, maintain vine health with balanced nutrition and proper irrigation"
    },
    "Grape___healthy": {
        "symptoms": "Leaves display uniform green color with no spots or discoloration. Fruit is properly colored and firm. Vines show vigorous growth with no wilting.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Irregular brown spots develop on leaves, often with concentric rings. Spots may have yellow halos. Affected leaves may drop prematurely. Severe infection causes defoliation.",
        "causes": "Fungal infection caused by Isariopsis clavispora (formerly Pseudocercospora vitis). The fungus survives on infected leaves and canes. Spores are spread by wind and rain splash. Disease development is favored by warm, wet conditions (21-27°C) with frequent rainfall. Dense canopies with poor air circulation create ideal conditions. The fungus penetrates leaf tissue and produces conidia that spread to healthy leaves.",
        "solutions": [
            "Apply fungicides containing mancozeb, captan, or myclobutanil",
            "Remove infected leaves to reduce spore production",
            "Improve air circulation through proper pruning and canopy management",
            "Apply fungicides preventively during wet weather periods",
            "Use copper fungicides for organic control",
            "Avoid overhead irrigation",
            "Apply treatments at 10-14 day intervals during wet weather"
        ],
        "prevention": "Maintain proper vine spacing and pruning for air circulation, remove and destroy infected leaves in fall, apply preventive fungicide sprays during wet spring weather, monitor weather forecasts and time applications before rain events, avoid overhead irrigation to keep foliage dry, use balanced fertilization to maintain vine health, plant resistant varieties when available, practice good sanitation by removing plant debris, scout vineyards regularly for early symptoms, maintain proper canopy management to reduce humidity"
    },
    # Orange
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Yellow shoots and asymmetrical blotchy mottling appear on leaves. Fruit becomes misshapen, lopsided, or aborted. Affected fruit is bitter and inedible. Trees show decline with reduced growth. Roots show brown discoloration.",
        "causes": "Bacterial infection caused by Candidatus Liberibacter spp. (primarily C. asiaticus). The bacteria are transmitted by Asian citrus psyllid (Diaphorina citri) and African citrus psyllid. The bacteria multiply in phloem tissue, blocking nutrient transport. Infected trees show zinc and copper deficiencies due to impaired nutrient uptake. The disease spreads rapidly through psyllid movement and infected nursery stock. No cure exists once trees are infected.",
        "solutions": [
            "Remove and destroy infected trees immediately to prevent spread",
            "Control psyllid vectors with insecticides containing imidacloprid or abamectin",
            "Apply systemic insecticides to nursery trees",
            "Use reflective mulches to repel psyllids",
            "Monitor trees regularly for early symptoms",
            "Use certified disease-free nursery stock only",
            "Implement area-wide psyllid control programs"
        ],
        "prevention": "Use certified disease-free nursery stock from trusted sources, implement strict quarantine measures for new plant material, control psyllid populations with regular insecticide applications, monitor trees weekly for symptoms and remove infected trees immediately, use reflective mulches around tree trunks to repel psyllids, plant windbreaks to reduce psyllid movement, maintain tree health with proper fertilization and irrigation, participate in area-wide management programs, avoid moving plant material from infected areas"
    },
    # Peach
    "Peach___Bacterial_spot": {
        "symptoms": "Small, dark, water-soaked circular spots appear on leaves, stems, and fruit. Spots are initially olive-green, then turn brown or black with yellow halos. Leaves may become yellowed and drop. Fruit spots become corky and raised.",
        "causes": "Bacterial infection caused by Xanthomonas campestris pv. pruni. The bacteria survive in infected buds, twigs, and fruit mummies. Bacteria spread through rain splash, irrigation water, and contaminated tools. Disease development requires warm, humid conditions (24-29°C) with extended wetness periods. Wounds from insects, hail, or frost provide entry points. The bacteria multiply rapidly in moist conditions and can spread throughout the orchard quickly.",
        "solutions": [
            "Apply copper-based bactericides every 10-14 days during wet weather",
            "Remove and destroy infected twigs and fruit mummies",
            "Prune trees to improve air circulation and reduce humidity",
            "Use resistant peach varieties like 'Redhaven' or 'Elberta'",
            "Apply streptomycin for severe infections (where permitted)",
            "Avoid overhead irrigation to keep foliage dry",
            "Apply preventive copper sprays during dormant season"
        ],
        "prevention": "Plant disease-resistant peach varieties, maintain proper tree spacing (15-20 feet) for air circulation, prune trees annually to remove infected wood and improve air flow, apply dormant copper sprays to kill overwintering bacteria, avoid overhead irrigation and wet foliage, remove and destroy fruit mummies and infected twigs, disinfect pruning tools between trees, monitor weather and apply preventive sprays before rain events, maintain tree health with balanced fertilization, use certified disease-free nursery stock"
    },
    "Peach___healthy": {
        "symptoms": "Leaves are smooth and green without spots. Fruit is firm, properly colored, and unblemished. Branches show no cankers or abnormal growth.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Pepper
    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Small, water-soaked circular spots with yellow halos appear on leaves and fruit. Spots develop brown or black centers with concentric rings. Leaves become yellowed and may drop. Fruit spots become corky with raised borders.",
        "causes": "Bacterial infection caused by Xanthomonas campestris pv. vesicatoria. The bacteria survive in infected seeds, plant debris, and weeds. Bacteria spread through rain splash, irrigation water, and contaminated tools. Disease thrives in warm, humid conditions (24-29°C) with extended leaf wetness. High nitrogen fertilization and dense planting increase susceptibility. The bacteria enter through stomata and wounds, multiplying rapidly in moist conditions.",
        "solutions": [
            "Apply copper-based bactericides every 7-10 days during wet weather",
            "Remove and destroy infected plant material immediately",
            "Improve air circulation by proper spacing (18-24 inches between plants)",
            "Use resistant pepper varieties with Bs genes",
            "Apply streptomycin or oxytetracycline for severe infections",
            "Switch to drip irrigation to keep foliage dry",
            "Apply preventive copper sprays at transplanting"
        ],
        "prevention": "Use certified disease-free seeds from reputable sources, practice 2-3 year crop rotation with non-solanaceous crops, space plants properly (18-24 inches) for air circulation, avoid overhead irrigation and wet foliage, apply balanced fertilization to avoid excessive nitrogen, disinfect tools and hands between plants, remove and destroy infected plants immediately, use reflective mulches to reduce bacterial spread, monitor fields regularly for early symptoms, plant resistant varieties with multiple resistance genes"
    },
    "Pepper,_bell___healthy": {
        "symptoms": "Leaves are uniformly green and smooth. Fruit is firm and properly colored. Plant shows vigorous growth with no spots or discoloration.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Potato
    "Potato___Early_blight": {
        "symptoms": "Dark, circular lesions with concentric rings (target-like appearance) develop on lower leaves first. Lesions have yellow halos and dark borders. Infected leaves become yellowed and drop. Stem lesions may also develop.",
        "causes": "Fungal infection caused by Alternaria solani. The fungus survives on infected plant debris, tubers, and seeds. Spores are spread by wind, rain splash, and irrigation water. Disease thrives in warm, humid conditions (24-29°C) with extended leaf wetness (more than 12 hours). High nitrogen fertilization, dense planting, and overhead watering create favorable conditions. The fungus penetrates through wounds or directly through leaf cuticle.",
        "solutions": [
            "Apply fungicides containing chlorothalonil or mancozeb every 7-10 days during humid weather",
            "Remove and destroy infected leaves immediately",
            "Improve air circulation by proper spacing (12-15 inches between plants)",
            "Use resistant potato varieties when available",
            "Apply copper fungicides as organic alternative",
            "Avoid overhead irrigation to keep foliage dry",
            "Apply preventive sprays at emergence and during tuber formation"
        ],
        "prevention": "Practice 2-3 year crop rotation with non-solanaceous crops, use certified disease-free seed potatoes, space plants properly (12-15 inches) for air circulation, avoid overhead irrigation and wet foliage, apply balanced fertilization to avoid excessive nitrogen, remove and destroy all plant debris after harvest, disinfect tools between plants, hill soil around plants to prevent tuber exposure, monitor fields regularly for early symptoms, plant resistant varieties when available"
    },
    "Potato___healthy": {
        "symptoms": "Leaves are uniform green with no spots or lesions. Stems are firm and upright. Plant shows vigorous growth.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Potato___Late_blight": {
        "symptoms": "Dark, water-soaked lesions appear on leaves and stems with a grayish appearance. White mold (mycelium) develops on the underside of affected leaves. Lesions expand rapidly, especially in cool, wet conditions. Infected tissues become soft and mushy. Tubers show brown, dry rot that starts at the skin and progresses inward. Affected tubers have a metallic taste when cooked.",
        "causes": "Oomycete infection caused by Phytophthora infestans (water mold, not true fungus). The pathogen survives in infected tubers, volunteer plants, and soil. Spores spread rapidly through wind-driven rain, irrigation water, and contaminated tools. Disease explodes in cool (16-21°C), humid conditions with extended leaf wetness. High nitrogen fertilization and dense planting increase susceptibility. The pathogen can survive in soil for years and spreads through infected seed potatoes.",
        "solutions": [
            "Apply fungicides containing mancozeb, chlorothalonil, or metalaxyl immediately at first symptoms",
            "Remove and destroy all infected plants and tubers - DO NOT compost",
            "Apply fungicides every 5-7 days during wet weather periods",
            "Use systemic fungicides like mefenoxam for severe outbreaks",
            "Harvest tubers immediately when disease appears to save crop",
            "Store harvested tubers in cool, dry conditions away from infected areas",
            "Apply copper-based fungicides as organic alternative",
            "Use reflective mulches to reduce spore germination"
        ],
        "prevention": "Use certified disease-free seed potatoes from reputable sources, practice 3-4 year crop rotation with non-solanaceous crops, plant resistant varieties when available, space plants properly for air circulation, avoid overhead irrigation, apply mulch to prevent soil splash, monitor weather forecasts and apply preventive fungicides before wet periods, fertilize with balanced NPK to avoid excessive nitrogen, hill soil around plants to prevent tuber exposure, destroy all plant debris and volunteer potatoes after harvest, disinfect tools and equipment between fields, plant early varieties to avoid peak disease season, maintain proper field drainage"
    },
    # Raspberry
    "Raspberry___healthy": {
        "symptoms": "Leaves display uniform green color with no spots or lesions. Berries are firm and properly colored. Canes are healthy without discoloration or damage.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Soybean
    "Soybean___healthy": {
        "symptoms": "Leaves are uniformly green with no spots, discoloration, or lesions. Pods are green and properly formed. Plant shows vigorous growth.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    # Squash
    "Squash___Powdery_mildew": {
        "symptoms": "White powdery coating develops on leaves, stems, and fruit. Infected leaves become distorted and curl. Fruit may be covered with white powder. Yellow patches appear on leaf surfaces. Severe infections cause leaves to wither and drop.",
        "causes": "Fungal infection caused by Podosphaera xanthii. The fungus survives as mycelium on infected plant debris and weeds. Spores are produced in warm, dry conditions and spread by wind. Disease development is favored by warm days (24-29°C) and cool nights with low humidity. Dense planting, poor air circulation, and high nitrogen levels increase susceptibility. The fungus grows on leaf surfaces without penetrating tissue, feeding on plant nutrients through haustoria.",
        "solutions": [
            "Apply sulfur-based fungicides or potassium bicarbonate sprays every 7-10 days",
            "Remove and destroy infected leaves to reduce spore production",
            "Improve air circulation through proper spacing (2-3 feet between plants)",
            "Use resistant squash varieties when available",
            "Apply neem oil or horticultural oils as organic alternatives",
            "Avoid overhead watering to keep foliage dry",
            "Apply treatments in early morning to avoid leaf burn"
        ],
        "prevention": "Plant resistant squash varieties, maintain proper plant spacing (2-3 feet) for air circulation, rotate crops with non-cucurbit crops for 2+ years, avoid overhead irrigation and wet foliage, apply balanced fertilization to avoid excessive nitrogen, prune plants to improve light penetration and reduce humidity, remove and destroy infected plant debris after harvest, use reflective mulches to reduce fungal growth, monitor plants regularly during warm weather, apply preventive sulfur sprays at first sign of disease"
    },
    # Strawberry
    "Strawberry___healthy": {
        "symptoms": "Leaves are vibrant green with no spots or lesions. Fruit is red and unblemished. Plant shows healthy, vigorous growth.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Strawberry___Leaf_scorch": {
        "symptoms": "Irregular brown spots with bright yellow or red halos develop on leaves. Spots are typically angular and scattered across the leaf surface. Severe infection causes premature leaf drop. Affected leaves dry up and become brittle.",
        "causes": "Fungal infection caused by Diplocarpon earlianum. The fungus survives on infected leaves and crowns. Spores are spread by rain splash and irrigation water. Disease development is favored by warm, humid conditions (21-27°C) with extended leaf wetness. Wet weather in spring and fall promotes infection. The fungus overwinters in infected plant debris and becomes active during wet periods. Dense planting and poor air circulation increase disease severity.",
        "solutions": [
            "Remove and destroy infected leaves immediately to reduce spore production",
            "Apply fungicides containing captan or myclobutanil every 7-10 days during wet weather",
            "Improve air circulation by proper plant spacing and runner thinning",
            "Avoid overhead irrigation to keep foliage dry",
            "Use resistant strawberry varieties when available",
            "Apply copper fungicides as organic alternative",
            "Apply preventive sprays during wet spring and fall periods"
        ],
        "prevention": "Plant resistant strawberry varieties, maintain proper plant spacing (12-18 inches) for air circulation, remove and destroy infected leaves and plant debris in fall, avoid overhead irrigation and wet foliage, apply mulch to prevent soil splash, thin runners regularly to improve air flow, disinfect tools between plants, apply preventive fungicide sprays during wet periods, monitor plants regularly for early symptoms, maintain good field drainage to reduce humidity"
    },
    # Tomato
    "Tomato___Bacterial_spot": {
        "symptoms": "Small, dark brown spots appear on leaves, stems, and green fruit with water-soaked centers. Spots have yellow halos and develop yellow borders. Lesions may coalesce into larger areas. Fruit spots become corky with raised edges as fruit ripens.",
        "causes": "Bacterial infection caused by Xanthomonas campestris pv. vesicatoria (and related Xanthomonas species). The bacteria survive in infected seeds, plant debris, and weeds. Bacteria spread through rain splash, irrigation water, and contaminated tools. Disease thrives in warm, humid conditions (24-29°C) with extended leaf wetness. High nitrogen fertilization and dense planting increase susceptibility. The bacteria enter through stomata and wounds, multiplying rapidly in moist conditions. Wind-driven rain spreads bacteria throughout fields quickly.",
        "solutions": [
            "Apply copper-based bactericides every 7-10 days during wet weather",
            "Remove and destroy infected plant material immediately",
            "Improve air circulation through proper spacing (2-3 feet between plants)",
            "Use certified disease-free seeds from reputable sources",
            "Apply streptomycin or oxytetracycline for severe infections",
            "Apply preventive copper sprays at transplanting",
            "Use reflective mulches to reduce bacterial spread"
        ],
        "prevention": "Use certified disease-free seeds treated with hot water or chemical treatments, practice 2-3 year crop rotation with non-solanaceous crops, space plants properly (2-3 feet apart) for air circulation, avoid overhead irrigation and wet foliage, apply balanced fertilization to avoid excessive nitrogen, disinfect tools and hands between plants, remove and destroy all infected plant debris after harvest, use reflective mulches to reduce bacterial splash, monitor fields regularly for early symptoms, plant resistant varieties when available, avoid working with plants when wet"
    },
    "Tomato___Early_blight": {
        "symptoms": "Concentric rings (target-like spots) appear on leaves starting on lower foliage. Lesions are dark brown with concentric bands and yellow halos. Stem lesions develop as elongated dark spots. Affected leaves turn yellow and drop. Fruit shows sunken, leathery spots with concentric rings. Severe infection causes defoliation and sunscald on exposed fruit.",
        "causes": "Fungal infection caused by Alternaria solani. The fungus survives on infected plant debris, seeds, and in soil. Spores are spread by wind, rain, irrigation water, and contaminated tools. Disease thrives in warm, humid conditions (24-29°C) with extended leaf wetness (more than 12 hours). High nitrogen fertilization, dense planting, and overhead watering create favorable conditions. The fungus penetrates through wounds or directly through leaf cuticle.",
        "solutions": [
            "Apply chlorothalonil, mancozeb, or copper-based fungicides every 7-10 days during humid weather",
            "Remove and destroy infected leaves immediately to reduce spore production",
            "Improve air circulation by proper spacing (2-3 feet between plants) and pruning",
            "Switch to drip irrigation to keep foliage dry",
            "Apply fungicides preventively at first sign of disease",
            "Use reflective mulches to reduce spore splash from soil",
            "Apply potassium bicarbonate (baking soda) sprays as organic alternative",
            "Remove volunteers and weed hosts around garden"
        ],
        "prevention": "Practice 2-3 year crop rotation with non-tomato crops, use certified disease-free seeds, space plants 2-3 feet apart for air circulation, avoid overhead watering, apply mulch to prevent soil splash, fertilize with balanced NPK to avoid excessive nitrogen, prune lower leaves to improve air flow, clean up all plant debris after harvest, disinfect tools between plants, plant resistant varieties when available, monitor weather and apply preventive sprays before rain, maintain proper soil drainage"
    },
    "Tomato___healthy": {
        "symptoms": "Leaves are uniformly green with no spots, lesions, or discoloration. Fruit is properly colored and firm. Plant shows vigorous growth with no wilting.",
        "causes": "Healthy plant",
        "solutions": [
            "Continue current care practices",
            "Maintain regular watering",
            "Apply balanced fertilizer",
            "Monitor for early signs of disease",
            "Maintain proper pruning"
        ],
        "prevention": "Continue preventive care and monitoring"
    },
    "Tomato___Late_blight": {
        "symptoms": "Water-soaked lesions appear on leaves, stems, and fruit. A white mold (mycelium) develops on the underside of affected leaves. Lesions expand rapidly, especially in cool, wet conditions. Affected tissues become dark and mushy. Green fruit develops brown lesions.",
        "causes": "Oomycete infection caused by Phytophthora infestans (water mold, not true fungus). The pathogen survives in infected plant debris, tubers, and soil. Spores spread through wind-driven rain, irrigation water, and contaminated tools. Disease explodes in cool (16-21°C), humid conditions with extended leaf wetness. The pathogen can survive in soil for years and spreads rapidly through wet weather. Both tomatoes and potatoes can be infected by the same pathogen.",
        "solutions": [
            "Apply fungicides containing chlorothalonil, mancozeb, or metalaxyl immediately at first symptoms",
            "Remove and destroy all infected plants immediately - DO NOT compost",
            "Apply fungicides every 5-7 days during wet, cool weather",
            "Use systemic fungicides like mefenoxam for severe outbreaks",
            "Improve air circulation by proper spacing and pruning",
            "Apply copper-based fungicides as organic alternative",
            "Harvest fruit immediately when disease appears"
        ],
        "prevention": "Practice 3-4 year crop rotation with non-solanaceous crops, use certified disease-free seeds and transplants, avoid planting near potatoes or in fields with late blight history, space plants properly (2-3 feet apart) for air circulation, avoid overhead irrigation and wet foliage, apply mulch to prevent soil splash, monitor weather forecasts and apply preventive fungicides before cool, wet periods, remove and destroy all plant debris after harvest, disinfect tools between plants, plant resistant varieties when available"
    },
    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow patches appear on the upper leaf surface. Olive-colored mold develops on the underside of affected leaves. Lesions become brown and dry. Severe infection causes leaves to wither and drop. Primarily affects lower leaves.",
        "causes": "Fungal infection caused by Passalora fulva (formerly Fulvia fulva). The fungus survives on infected plant debris and seeds. Spores are spread by wind and water splash. Disease thrives in warm (21-27°C), humid conditions with poor air circulation. High humidity (above 85%) and extended leaf wetness favor development. The fungus grows on leaf surfaces and penetrates through stomata. Greenhouse-grown tomatoes are particularly susceptible due to controlled humidity.",
        "solutions": [
            "Improve air circulation by proper spacing and pruning lower leaves",
            "Reduce greenhouse humidity below 85% using ventilation and heating",
            "Apply fungicides containing chlorothalonil or mancozeb every 7-10 days",
            "Remove and destroy infected leaves immediately",
            "Use resistant tomato varieties when available",
            "Apply copper fungicides as organic alternative",
            "Increase temperature slightly to reduce humidity"
        ],
        "prevention": "Maintain proper plant spacing (2-3 feet) for air circulation, prune lower leaves to improve ventilation, control greenhouse humidity below 85%, avoid overhead irrigation and wet foliage, use fans for air circulation in greenhouses, apply balanced fertilization to maintain plant health, remove and destroy infected plant debris, disinfect greenhouse structures between crops, use certified disease-free seeds, monitor humidity levels regularly, plant resistant varieties when available"
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small, circular spots with dark borders appear on leaves. Centers of lesions are grayish-white with dark concentric rings. Spots develop yellow halos. Black fruiting bodies (pycnidia) are visible in the center of lesions as small black dots. Severe infections cause defoliation.",
        "causes": "Fungal infection caused by Septoria lycopersici. The fungus survives on infected plant debris and seeds. Spores are spread by wind, rain splash, and irrigation water. Disease thrives in warm, humid conditions (20-25°C) with extended leaf wetness (more than 12 hours). The fungus penetrates through stomata and produces pycnidia containing spores. Wet weather in spring and fall promotes rapid spread. The disease is seed-borne and can survive in soil for several years.",
        "solutions": [
            "Apply fungicides containing chlorothalonil or mancozeb every 7-10 days during wet weather",
            "Remove and destroy infected leaves immediately to reduce spore production",
            "Improve air circulation by proper spacing (2-3 feet between plants)",
            "Switch to drip irrigation to keep foliage dry",
            "Apply copper fungicides as organic alternative",
            "Use resistant tomato varieties when available",
            "Apply preventive sprays at transplanting and after rainfall"
        ],
        "prevention": "Practice 2-3 year crop rotation with non-tomato crops, use certified disease-free seeds from reputable sources, space plants properly (2-3 feet apart) for air circulation, avoid overhead irrigation and wet foliage, remove and destroy all plant debris after harvest, apply mulch to prevent soil splash, disinfect tools between plants, apply preventive fungicide sprays during wet periods, monitor plants regularly for early symptoms, plant resistant varieties when available, maintain proper field drainage"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Fine, light stippling appears on leaves, creating a yellow-bronzed appearance. Fine webbing develops on the undersides and between leaves. Severe infestations cause leaves to turn brown and drop. Mites are visible as tiny moving dots under magnification.",
        "causes": "Spider mite infestation caused by Tetranychus urticae (two-spotted spider mite). Mites are not insects but arachnids related to spiders and ticks. They thrive in hot, dry conditions (27-32°C) with low humidity. Mites pierce leaf cells and suck out contents, causing stippling damage. They reproduce rapidly (7-10 day life cycle) and can develop resistance to pesticides. Drought stress and dusty conditions increase plant susceptibility. Mites spread through wind, on clothing/tools, and by crawling between plants.",
        "solutions": [
            "Apply insecticidal soap or neem oil sprays every 3-5 days for 2-3 weeks",
            "Introduce predatory mites (Galendromus occidentalis) as biological control",
            "Increase humidity around plants with overhead misting",
            "Use miticides containing abamectin or spiromesifen for severe infestations",
            "Remove and destroy heavily infested leaves",
            "Apply horticultural oils to smother mites",
            "Use systemic insecticides like imidacloprid for long-term control"
        ],
        "prevention": "Maintain adequate humidity (50-70%) around plants to discourage mite development, avoid drought stress by providing consistent watering, regularly rinse plants with water to remove dust and mites, monitor plants weekly for early stippling symptoms, avoid broad-spectrum insecticides that kill beneficial predators, plant mite-resistant tomato varieties when available, maintain proper plant spacing for air circulation, use reflective mulches to deter mites, introduce beneficial insects early in the season, avoid planting near dusty roads or areas with high mite pressure"
    },
    "Tomato___Target_Spot": {
        "symptoms": "Concentric rings develop on leaves and fruit, creating a target-like appearance. Lesions are dark brown with light centers. Small black fruiting bodies are visible in the center. Infected leaves become yellowed and may drop. Fruit spots are sunken with concentric rings.",
        "causes": "Fungal infection caused by Corynespora cassiicola. The fungus survives on infected plant debris and seeds. Spores are spread by wind, rain splash, and irrigation water. Disease thrives in warm, humid conditions (25-30°C) with extended leaf wetness. The fungus produces abundant spores that spread rapidly. It affects many crops including tomatoes, peppers, and cucumbers. High temperatures and humidity favor rapid disease development and spread.",
        "solutions": [
            "Apply fungicides containing chlorothalonil or azoxystrobin every 7-10 days during humid weather",
            "Remove and destroy infected plant material immediately",
            "Improve air circulation through proper spacing and pruning",
            "Use resistant tomato varieties when available",
            "Apply copper fungicides as organic alternative",
            "Apply preventive sprays during high humidity periods",
            "Increase fungicide frequency during warm, humid weather"
        ],
        "prevention": "Practice 2-3 year crop rotation with non-solanaceous crops, use certified disease-free seeds and transplants, maintain proper plant spacing (2-3 feet) for air circulation, avoid overhead irrigation and wet foliage, monitor weather forecasts for high humidity periods, apply preventive fungicide sprays before humid weather, remove and destroy all plant debris after harvest, disinfect tools between plants, use balanced fertilization to maintain plant health, plant resistant varieties when available"
    },
    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Mottled yellow and green patterns appear on leaves in an irregular mosaic pattern. Leaves become crinkled and distorted. Growth is stunted. Fruit may develop spots or patches. Some leaves may show severe necrosis.",
        "causes": "Viral infection caused by Tobacco mosaic virus (TMV). The virus survives in infected plant debris, seeds, and soil for years. It spreads through contaminated tools, hands, and clothing. Aphids and other insects can transmit the virus while feeding. The virus enters through wounds and abrasions on plant tissue. TMV is extremely stable and can survive in dried plant sap for decades. High temperatures (above 32°C) can inactivate the virus, but it thrives in cooler conditions.",
        "solutions": [
            "Remove and destroy infected plants immediately to prevent spread",
            "Use certified virus-free seeds from reputable sources",
            "Control aphid vectors with insecticidal soaps or neem oil",
            "Disinfect all tools, hands, and equipment with 10% bleach solution",
            "Plant resistant tomato varieties with Tm genes",
            "Avoid smoking near plants (TMV can be transmitted from tobacco)",
            "Use hot water seed treatment (50°C for 10 minutes) for seed disinfection"
        ],
        "prevention": "Use certified virus-free seeds tested for TMV, practice strict sanitation by disinfecting tools and hands between plants, control aphid populations with regular monitoring and organic insecticides, avoid planting near tobacco plants or in areas with TMV history, plant resistant varieties with multiple Tm genes, maintain proper plant spacing to reduce spread, remove weeds that can harbor the virus, avoid working with wet plants to prevent mechanical transmission, use row covers to exclude insect vectors, test soil and plant material before planting"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Leaves curl upward and become stunted and yellowed. Lower leaves show more severe symptoms. Plant growth is severely stunted with shortened internodes. Leaf edges appear serrated. Fruit is undersized and stays green. Plant wilts despite adequate water.",
        "causes": "Viral infection caused by Tomato yellow leaf curl virus (TYLCV), transmitted by the sweet potato whitefly (Bemisia tabaci). The virus is persistent in the whitefly vector and spreads rapidly once introduced. The virus replicates in phloem tissue, causing severe physiological disruption. Infected plants produce more virus particles that attract more whiteflies. The disease spreads exponentially in warm climates where whitefly populations are high. No seed transmission occurs.",
        "solutions": [
            "Remove and destroy infected plants immediately to reduce virus source",
            "Control whitefly vectors with systemic insecticides like imidacloprid",
            "Use resistant tomato varieties with Ty genes",
            "Apply insecticidal soaps or neem oil for organic control",
            "Use reflective mulches to repel whiteflies",
            "Install yellow sticky traps to monitor and reduce whitefly populations",
            "Apply kaolin clay sprays to deter whiteflies"
        ],
        "prevention": "Use resistant tomato varieties with multiple Ty genes for broad-spectrum resistance, monitor whitefly populations with yellow sticky traps, apply systemic insecticides preventively in high-risk areas, use reflective mulches (silver or aluminum) around plants to repel whiteflies, maintain proper plant spacing for air circulation, remove weeds that attract whiteflies, avoid planting near infected crops or in areas with TYLCV history, use row covers to exclude whiteflies during early growth, implement area-wide whitefly control programs, quarantine new plant material"
    }
}

if __name__ == '__main__':
    # ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation',
        shuffle=False
    )

    class_names = list(train_gen.class_indices.keys())
    print(f"Found {len(class_names)} classes: {class_names}")

    # MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.3),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(len(class_names), activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        callbacks.ModelCheckpoint(
            'detector_disease.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    # Fine-tuning
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history_fine = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // batch_size,
        epochs=20,
        validation_data=val_gen,
        validation_steps=val_gen.samples // batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    # Evaluate model
    val_gen.reset()
    val_steps = val_gen.samples // batch_size
    val_preds = model.predict(val_gen, steps=val_steps, verbose=1)
    y_pred_classes = np.argmax(val_preds, axis=1)
    y_true_classes = val_gen.classes[:val_steps * batch_size]

    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    model.save('models/detector_disease.keras')

    # Ploting history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')

    # Final accuracy
    val_loss, test_accuracy = model.evaluate(val_gen, steps=val_gen.samples // batch_size, verbose=1)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
