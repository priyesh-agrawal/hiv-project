#The script is written to do the metabolomics analysis of the compounds using MetaboAnalyst API
import requests
import json
import csv
url = "https://www.xialab.ca/api/mapcompounds"


METABOLITES_1 = ["9-HpODE","Abscisic acid","(2R)-2,3-Dihydroxypropanoic acid","(2R)-2,3-Dihydroxypropanoic acid","(2R)-2,3-Dihydroxypropanoic acid","(5Z)-3-aminonon-5-enoic acid","Linolenic Acid","1,5-Anhydro-D-glucitol","1,7-Dimethyluric acid","10-Hydroxydecanoic acid","11(Z),14(Z)-Eicosadienoic acid","11(Z)-Eicosenoic acid","13,14-Dihydro-15-keto Prostaglandin A2","13-hydroxy-6-(1-hydroxyethyl)-12,14-dimethyl-3-methylidene-15-nonyl-9-(propan-2-yl)-1-oxa-4,7,10-triazacyclopentadecane-2,5,8,11-tetrone","16-Hydroxyhexadecanoic acid","1-Methylhistidine","1-Methylnicotinamide","1-Methylxanthine","2-(3,4-dihydroxyphenyl)-3,5,7-trihydroxy-6-methyl-4H-chromen-4-one","2,4-Dihydroxybenzoic acid","2,4-Dihydroxybenzoic acid","2,6-Di-tert-butyl-1,4-benzoquinone","2-Anisic acid","2-Furoylglycine","2-Hydroxyhippuric acid","2-Hydroxyhippuric acid","2-Hydroxyhippuric acid","2-Hydroxyhippuric acid (mzCloud ID 4)","2-Hydroxyvaleric acid","2-Oxindole","2-Phenylphenol","2-Phospholactate","3-Hydroxyanthranilic acid","3-Hydroxybutyric acid","3-Indoxyl sulphate","3-Methoxyphenylacetic acid","3-Methylhistidine","3-methylphenylacetic acid","3-Methylxanthine","3-Phosphoglyceric acid","3-Pyridinol","4-Anisic acid","4-Hydroxyproline","4-Methylphenol","4-Oxoproline","4-Pyridoxic acid","5-acetyl-2,6-dimethyl-1,2,3,4-tetrahydropyridin-4-one","7-Methylguanine","8Z,11Z,14Z-Eicosatrienoic acid","9-Oxo-ODE","Adenosine monophosphate","Adenosine triphosphate","ADP","Aminoadipic acid","Arachidic Acid","Arachidonic acid","Arecoline","Azelaic acid","Betaine","Bilirubin","Bilirubin","Bilirubin","Caffeine","Caprylic acid","Choline","cis-Aconitic acid","Citraconic Acid","Citric acid","Citrulline","Cotinine","Creatine","Creatinine","D-(-)-Quinic acid","D-(-)-Quinic acid","D-2-Hydroxyglutaric acid","Decanoic acid","Decanoylcarnitine","Decanoylcarnitine","Deoxycholic Acid","Deoxycholic Acid","Deoxycholic Acid","D-Erythro-sphingosine 1-phosphate","D-Glucose","DHAP/GAP","Dimethylglycine","DL-4-Hydroxyphenyllactic acid","DL-Stachydrine","DL-?-Aminocaprylic acid","Docosahexaenoic Acid","D-Ribose 5-phosphate","D-Saccharic acid","D-?-Hydroxyglutaric acid","D-?-Tocopherol","Ethyl myristate","Fisetin","Glycerol","Glycerol 3-phosphate","Glycerophospho-N-palmitoyl ethanolamine","Glycerophospho-N-palmitoyl ethanolamine","Glycine","Glycodeoxycholic Acid (hydrate)","Glycoursodeoxycholic acid","Guanine","Guanine","Guvacoline","Guvacoline","Hexanoylcarnitine","Hippuric Acid","Hydroxyisocaproic acid","Hypoxanthine","Imidazoleacetic acid","Imidazolelactic acid","Indole-3-acetic acid","Indole-3-lactic acid","Indole-3-pyruvic acid","Inosine","Inosinic acid","Isoquinoline","Isorhamnetin","L(-)-Carnitine","L-Acetylcarnitine","L-Alanine","L-Arginine","L-Ascorbic acid 2-sulfate","L-Ascorbic acid 2-sulfate","L-Asparagine","L-Aspartic acid","L-Carnitine","L-Cysteine-S-sulfate","L-Cystine","L-Ergothioneine","Leucylproline","Leucylproline","Leucylproline","L-Glutamic acid","L-Glutamine","L-Glyceric acid","L-Histidine","L-Iditol","Linoleic acid","L-Isoleucine","L-Kynurenine","L-Lactic acid","L-Leucine","L-Lysine","L-Methionine","L-Phenylalanine","L-Proline","L-Serine","L-Threonine","L-Tryptophan","L-Tyrosine","L-Valine","Malic acid","Mesaconic acid","Meso-erythritol","Methionine sulfoxide","Methyl indole-3-acetate","Methylcysteine","Methylimidazoleacetic acid","Methylimidazoleacetic acid","Myristic acid","N-(2-hydroxyphenyl)acetamide","N-(2-hydroxyphenyl)acetamide","N-(2-hydroxyphenyl)acetamide","N4-Acetylcytidine","N6,N6,N6-Trimethyl-L-lysine","N6-Acetyl-L-lysine","N6-Acetyl-L-lysine","N-Acetylglutamic acid","N-Acetylglutamine","N-Acetyl-L-alanine","N-Acetyl-L-aspartic acid","N-Acetyl-L-carnosine","N-Acetyl-L-carnosine","N-Acetylneuraminic acid","N-Acetylornithine","Nicotinamide","N-Isovalerylglycine","N-Isovalerylglycine","N-Phenylacetylglutamine","N-Phenylacetylglutamine","N-?-L-Acetyl-arginine","O-Acetylserine","Oleic acid","Oleoyl-L-?-lysophosphatidic acid","Oleoyl-L-?-lysophosphatidic acid","O-Phosphorylethanolamine","Ornithine","Orotic acid","Orotidine","Oxoamide","Oxoglutaric acid","Palmitoleic acid","Palmitoyl sphingomyelin","Palmitoyl sphingomyelin","Palmitoylcarnitine","Palmitoylcarnitine","Pantothenic acid","Phenyllactic acid","Phenylpyruvic acid","Phosphoenolpyruvic acid","Pipecolic acid","Piperine","PLK","Prolylglycine","Propionylcarnitine","Propionylcarnitine","Pyroglutamic acid","Pyruvic acid","Quinolinic acid","S-Adenosylhomocysteine","Sarcosine","S-Methyl-L-cysteine-S-oxide","Succinic acid","Succinic semialdehyde","Sucrose","Symmetric dimethylarginine","Taurine","Taurochenodeoxycholic acid","Theophylline","Theophylline","Theophylline","trans-10-Heptadecenoic Acid","trans-3-Hydroxycotinine","Trigonelline","Trimethylamine N-oxide","Tropine","Uric acid","Uridine","Valylproline","Xanthine","Xylitol","Ribono-1,4-lactone"]

payload = "{\n\t\"queryList\": \""+format(";".join(METABOLITES_1))+"\",\n\t\"inputType\": \"name\"\n}"

headers = {
    'Content-Type': "application/json",
    'cache-control': "no-cache",
    }

response = requests.request("POST", url, data=payload, headers=headers)

if response.status_code == 200:
    FH = open("metabo_out.tsv", "w", newline="")
    writer = csv.writer(FH, delimiter="\t")
    data  = json.loads(response.text)

    for i in range(len(METABOLITES_1)):
        #print(METABOLITES_1[i], data["Query"][i], data['Match'][i], data['HMDB'][i], data['PubChem'][i], data['ChEBI'][i], data['KEGG'][i], data['METLIN'][i], data['SMILES'][i], data['Comment'][i], sep="\t")
        out_row = [METABOLITES_1[i], data["Query"][i], data['Match'][i], data['HMDB'][i], data['PubChem'][i], data['ChEBI'][i], data['KEGG'][i], data['METLIN'][i], data['SMILES'][i], data['Comment'][i]]
        writer.writerow(out_row)
    FH.close()
#print(data.keys())
else:
    print(f'Error in fetching the data! ({response.status_code})')
