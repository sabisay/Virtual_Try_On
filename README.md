# Virtual_Try_On

1.	Giriş:</br>
Bu proje, derin öğrenme tabanlı modeller kullanarak insan vücudu segmentasyonu gerçekleştirmeyi amaçlamaktadır. Projenin temel hedefi, vücut bölümlerini doğru bir şekilde ayırarak sanal giydirme gibi uygulamalara zemin hazırlamaktır.
Başlangıçta farklı kaynaklardan dört farklı repo incelenmiş ve mevcut çözümler analiz edilmiştir. Bu analizler sonucunda, kod yazma girişiminde bulunmuş ancak çeşitli zorluklarla karşılaşılmıştır. Bu zorlukların üstesinden gelmek adına, ChatGPT, Claude ve Cursor AI gibi yapay zeka destekli araçlar kullanılarak kod geliştirme süreci optimize edilmiştir.
İlk aşamada yazılan kod, verimlilik ve segmentasyon doğruluğu açısından çeşitli sınırlamalar içermekteydi. Bu sebeple, “Older” klasörüne atılıp, geliştirilmiş bir versiyon oluşturularak “Enhaced” klasöründe daha hassas segmentasyon ve daha iyi görselleştirme sonuçları elde edilmiştir. </br>
İncelenen Repolar:</br>
•	https://github.com/cuiaiyu/dressing-in-order </br>
•	https://github.com/lllyasviel/Fooocus</br>
•	https://github.com/HumanAIGC/OutfitAnyone</br>
•	https://github.com/levindabhi/cloth-segmentation</br>

2.	Kullanılan Kütüphaneler:</br>
•	cv2 (OpenCV):Görüntü işleme ve maskeleme işlemleri için kullanıldı.</br>
•	numpy: Sayısal işlemler ve dizi manipülasyonları için kullanıldı.</br>
•	os: Dosya işlemleri ve dizin yönetimi için kullanıldı.</br>
•	ultralytics (YOLO): Derin öğrenme tabanlı nesne tespiti ve segmentasyon için kullanıldı.</br>
•	PIL (Pillow): Görüntü dosyalarının işlenmesi ve dönüştürülmesi için kullanıldı.</br>
•	torch: Derin öğrenme modellerini çalıştırmak için kullanıldı.</br>
•	mediapipe: İnsan vücut noktalarını tespit etmek ve analiz etmek için kullanıldı.</br>
•	scipy.interpolate (Rbf): Görüntü verileri üzerinde interpolasyon işlemleri için kullanıldı.</br>
•	jax ve jax.numpy: Hızlı sayısal hesaplamalar yapmak için kullanıldı.</br>
•	keras: Derin öğrenme modellerini eğitmek ve kullanmak için kullanıldı.</br>
3.	Kod Açıklaması:</br>
Ana Klasör: Virtual_Try_On</br>
-	Enhanced: Bir Segmentation ve Warping işlemi içeren deneme.</br>
o	Outputs: tişört ile warp edilmiş avatar çıktıları burada toplanmıştır.</br>
o	Segmentation: Maskelenmiş Segmentler burada toplanmıştır. </br>
o	EnhancedSegmentation.py: Bu kod, YOLO tabanlı model ile geliştirilmiş insan vücut segmentasyonu sağlar.</br>
	Önişleme: Görüntü RGB formatına dönüştürülür, orijinal oran korunarak yeniden boyutlandırılır.</br>
	Segmentasyon: Model çıktısı işlenerek tek kanallı maske elde edilir, hata ayıklama mekanizmaları eklenmiştir.</br>
	İyileştirme: Gaussian Blur, Otsu thresholding ve morfolojik işlemler ile maske daha pürüzsüz ve hatasız hale getirilir.</br>
	Vücut Çıkartma: Maske kullanılarak arka plan temizlenir, sadece insan vücudu izole edilir.</br>
	Sonuçların Kaydedilmesi: Çıktılar zaman damgalı olarak kaydedilir, böylece versiyon takibi sağlanır.</br>

o	EnhancedWarping.py: Bu kod, MediaPipe ve Thin Plate Spline (TPS) kullanarak giysi deformasyonunu optimize eder.</br>
	Gelişmiş Landmark Algılama: MediaPipe ile omuz ve kalça noktaları tespit edilir, başarısız olursa görüntü konturları kullanılarak tahmini noktalar oluşturulur.</br>
	TPS Warping: Giysi, kaynak ve hedef noktalar arasındaki esnek dönüşümle vücuda uyarlanır.</br>
	Maskeli Alfa Karışımı: Giysi maskesi Gaussian Blur ile yumuşatılır, alfa karışımı ile daha doğal bir görüntü elde edilir.</br>
	Geliştirilmiş Hata Kontrolü: Landmark algılama hatalarını ele alarak süreci daha dayanıklı hale getirir.</br>

-	Older: Bir Segmentation, Warping ve Blending içeren deneme. İki dosya da ayrı ve bağımsız çalışmaktadır.</br>

o	BlendingAndRefinement.py: Bu kod, seamless cloning tekniğini kullanarak giysi ve insan vücudunu daha doğal bir şekilde birleştirir.</br>
	Maske İşleme: Maske, vücut görüntüsüyle aynı boyuta getirilir ve renk kanalları uyarlanır.</br>
	Seamless Cloning: OpenCV’nin cv2.seamlessClone fonksiyonu ile giysi ve vücut, doğal geçişler sağlanarak harmanlanır.</br>
	Otomatik Dosya Yönetimi: Çıktılar için klasör oluşturur ve aynı dosya adına sahip bir çıktı varsa isimlendirmeyi otomatik olarak günceller.</br>
	Hata Kontrolü: Eksik veya yanlış yolları algılayarak işlemi güvenli hale getirir.</br>

o	BodySegmentation.py: Bu kod, U2Net tabanlı bir model kullanarak insan vücudunu görüntülerden ayırır.</br>
	Önişleme: Görüntü okunur, RGB formatına dönüştürülür ve model girişine uygun hale getirilir.</br>
	Segmentasyon: U2Net modeli ile insan vücudu maskesi oluşturulur ve işlenerek tek kanallı hale getirilir.</br>
	Son İşlemler: Gaussian Blur ve morfolojik işlemler ile maske iyileştirilir, vücut bölgesi ayrılarak kaydedilir.</br>

o	ClothingWarping.py: Bu kod, giysi görsellerini vücut şekline uyarlamak için kullanılır.</br>
	Anahtar Nokta Tespiti: Vücut üzerindeki omuz, bel gibi noktalar belirlenir.</br>
	Warping İşlemi: Thin Plate Spline (TPS) ve optik akış yöntemleriyle giysi vücuda esnetilerek uyarlanır.</br>
	Son İşlemler: Alpha blending ve yumuşatma uygulanarak giysi daha doğal görünüm kazanır.</br>
o	SettingU2Net.py: JAX ile modelin entegrasyonunda kullanılmıştır.</br>

o	Masks: BodySegmentation.py sonucu oluşan maskeler burada toplanmıştır.</br>

-	Shirts: Arkaplan içerikli, arkaplan içeriksiz, koyu renkli, açık renkli gibi ayrıştırıcı özelliklere sahip 3 adet tişört resmi içerir. Hem Enhanced hem de Older kodunda bu tişörtler denenmiştir.</br>

-	venv: Virtual Environment ortamıdır. Numpy, Tensorflow gibi kütüphanelerin çeşitli fonksiyonlarının uyumluluk sorunu çıkarmaması adına Python’un en yeni sürümü olmayan, ancak en stabil sürümü olan python 3.10.0 ile oluşturulup, gerekli kütüphaneler içerisine import edilmiştir.</br>


-	Person.jpg: Segmente edilemye çalışılan avatar örneğidir.</br>

-	Yolo8n-seg.pt: YOLO kütüphanesini kullanabilmek için gerekli dosyadır.</br>
