V1 del fine-tuning.

-> DATSET complet de train i val de TinyImagenet.
   train = 500 img x 200 classes = 100.000 imatges
   val = 10.000 imatges, classificades gràcies al arxiu val_annotations

   -> GASTO EN RAM 110.000 imatges en tamany 64x64 ~aprox. 5 GB
   
-> Permet, amb la v1, entrenar un model que accepta com a entrada imateges de TinyImagenet sense
    redimensionar, de 64x64. Rendiment molt pobre, quasi random 0.5% d'accerts.
    No testejat amb profunditat

-> v2 similar però amb un codi inspirat en un trobat a la web medium.com, també genera un model
   però amb rendiment igual de roïn, 0.5% d'accerts.
   testejat amb profunditat, carrega la RAM en qualquier operació que involucre imatges de mes de
   64x64