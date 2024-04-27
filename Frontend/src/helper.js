const map = new Map();
map.set("Fungal infection", "Dermatologist");
map.set("Psoriasis", "Dermatologist");
map.set("Impetigo", "Dermatologist");
map.set("Chicken pox", "Dermatologist");
map.set("Acne", "Dermatologist");
map.set("Allergy", "Immunologist");
map.set("AIDS", "Immunologist");
map.set("Pneumonia", "Immunologist");
map.set("Bronchial Asthma", "Immunologist");
map.set("hepatitis A", "Hepatologist");
map.set("Hepatitis B", "Hepatologist");
map.set("Hepatitis C", "Hepatologist");
map.set("Hepatitis D", "Hepatologist");
map.set("Hepatitis E", "Hepatologist");
map.set("Alcoholic hepatitis", "Gastroenterologist");
map.set("Typhoid", "Gastroenterologist");
map.set("Jaundice", "Gastroenterologist");
map.set("Gastroenteritis", "Gastroenterologist");
map.set("Peptic ulcer disease", "Gastroenterologist");
map.set("Chronic cholestasis", "Gastroenterologist");
map.set("GERD", "Gastroenterologist");
map.set("Hypothyroidism", "Endocrinologist");
map.set("Hyperthyroidism", "Endocrinologist");
map.set("Hypoglycemia", "Endocrinologist");
map.set("Malaria", "Infectious Disease Specialist");
map.set("Tuberculosis", "Infectious Disease Specialist");
map.set("Dengue", "Endocrinologist");
map.set("Common Cold", "Endocrinologist");
map.set("Osteoarthritis", "Orthopedic Surgeon");
map.set("Arthritis", "Orthopedic Surgeon");
map.set("Cervical spondylosis", "Orthopedic Surgeon");
map.set("Paralysis (brain hemorrhage)", "Neurologist");
map.set("Migraine", "Neurologist");
map.set("Hypertension", "Neurologist");
map.set("(vertigo) Paroxysmal Positional Vertigo", "Neurologist");
map.set("Heart attack", "Cardiologist");
map.set("Varicose veins", "Phlebologist");
map.set("Urinary tract infection", "Urologist");
map.set("Dimorphic hemorrhoids(piles):", "Proctologist");

export function call_me(params) {
  const arr = [];
  for (let i = 0; i < params.length; i++) {
    arr.push(map.get(params[i]));
  }
  return arr;
}