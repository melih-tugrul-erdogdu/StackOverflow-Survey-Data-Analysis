# 📊 Unveiling the Realities of the Software Industry: AI, Work Models, and Salaries
*(Descriptive and Inferential Analysis based on the 2024 Stack Overflow Developer Survey)*

## 🚀 Project Overview
Does using AI tools actually boost a developer's salary? Does a university degree still matter in a world full of self-taught coders? 

Welcome to a comprehensive statistical deep-dive into the modern software engineering workforce. Using a carefully cleaned sample of **20,754 developers** from the 2024 Stack Overflow Developer Survey, this project cuts through the industry hype. We explore the genuine financial impacts of educational backgrounds, professional experience, work environments, and the rapidly growing adoption of Artificial Intelligence.

## 🎯 Key Research Questions
We set out to demystify the market by answering five critical questions:
* **RQ1:** Does formal university education truly give developers a head start in professional experience compared to alternative paths?
* **RQ2:** Are senior developers embracing AI coding tools faster, or are they sticking to their traditional workflows?
* **RQ3:** *The Million Dollar Question:* Does active AI integration actually lead to higher annual compensation?
* **RQ4:** How do different work environments (Remote, Hybrid, In-person) realistically shape salary distributions?
* **RQ5:** Which programming language ecosystems are the quickest to adopt AI?

## 🔬 Methodology: Behind the Numbers
This project goes beyond simple bar charts. We transitioned from observational trends to rigorous, statistically significant conclusions using a robust analytical pipeline:
* **Exploratory Data Analysis (EDA):** Uncovering the shape of the data through visualizations and descriptive statistics (mean, median, variance).
* **Hypothesis Testing:** Applying Independent Samples T-Tests (Welch's) and One-Way ANOVA to validate observed group differences.
* **Multiple Linear Regression (OLS):** The core of our study. We built an OLS model to isolate the true financial impact of AI adoption by strictly controlling for confounding variables like seniority and formal education.

## 💡 Main Findings
* 🎓 **The True Predictors of Salary:** Experience and formal education remain the undisputed kings of compensation. Holding other factors constant, a university degree adds an average premium of **$13,640**, while every single year of professional experience boosts the annual salary by approximately **$2,498**.
* 🤖 **The "AI Salary Paradox":** Here is where the data gets fascinating. Initially, a simple T-Test suggested that developers *not* using AI earned more. However, our regression analysis proved this was a statistical illusion caused by seniority—senior developers earn more overall but are adopting AI at a slower rate. Once we control for experience, the effect of AI usage on salary becomes completely insignificant (**p = 0.587**). Simply put: AI doesn't pay more; experience does.
* 🏠 **The Remote Work Premium:** Working from home pays off. Fully remote developers enjoy an average salary premium of **$11,120** over hybrid roles, while strict in-person positions take a noticeable financial hit.

📄 **Project Report:** [Stack Overflow Survey Analysis Report (PDF)](StackOverflow_Survey_Analysis_Report.pdf)

---

# 📊 Yazılım Sektörünün Gerçekleri: YZ Kullanımı, Çalışma Modelleri ve Maaş Analizi
*(2024 Stack Overflow Geliştirici Anketi Üzerine Betimsel ve Çıkarımsal Analiz)*

## 🚀 Proje Hakkında
Yapay zeka araçlarını kullanmak bir yazılımcının maaşını gerçekten artırır mı? Alaylı geliştiricilerin hızla arttığı bir dünyada üniversite diploması hala önemli mi?

Modern yazılım mühendisliği iş gücünün derinliklerine indiğim bu istatistiksel analize hoş geldiniz! 2024 Stack Overflow Geliştirici Anketi'nden özenle temizlenmiş **20.754 kişilik** bir veri seti kullanarak sektördeki popüler efsaneleri mercek altına alıyoruz. Bu proje; eğitim geçmişinin, mesleki deneyimin, çalışma modellerinin ve hızla yayılan Yapay Zeka kullanımının maaşlar üzerindeki gerçek finansal etkilerini araştırmaktadır.

## 🎯 Temel Araştırma Soruları
Sektördeki sis perdesini aralamak için şu beş kritik sorunun peşine düştüm:
* **RQ1:** Üniversite mezunları ile alternatif yollardan gelen geliştiriciler arasında sektörel deneyim birikimi açısından anlamlı bir fark var mı?
* **RQ2:** Kıdemli geliştiriciler YZ tabanlı kodlama araçlarını daha mı hızlı benimsiyor, yoksa geleneksel yöntemlere bağlı mı kalıyorlar?
* **RQ3:** *Projenin En Önemli Sorusu:* Aktif olarak YZ kullananlar, kullanmayanlara göre gerçekten daha fazla mı kazanıyor?
* **RQ4:** Farklı çalışma ortamları (Uzaktan, Hibrit, Ofis) maaş dağılımlarını gerçekte nasıl şekillendiriyor?
* **RQ5:** Hangi programlama dili ekosistemleri YZ araçlarını benimsemede başı çekiyor?

## 🔬 Metodoloji: Sayıların Arkasındaki Matematik
Bu proje sadece basit grafiklerden ibaret değil. Gözlemsel trendlerden istatistiksel olarak kanıtlanmış sonuçlara ulaşmak için güçlü bir analitik süreç işlettim:
* **Keşifçi Veri Analizi (EDA):** Verinin doğasını anlamak için görselleştirmeler ve betimsel istatistikler (ortalama, medyan, varyans) kullanıldı.
* **Hipotez Testleri:** Gözlemlenen grup farklılıklarının tesadüfi olmadığını kanıtlamak için Bağımsız Örneklemler T-Testi (Welch) ve Tek Yönlü ANOVA uygulandı.
* **Çoklu Doğrusal Regresyon (OLS):** Çalışmamızın kalbi. Kıdem ve eğitim gibi kafa karıştırıcı değişkenleri kontrol altında tutarak, YZ kullanımının maaş üzerindeki saf finansal etkisini izole etmek için bir OLS modeli inşa ettik.

## 💡 Temel Bulgular
* 🎓 **Maaşın Gerçek Belirleyicileri:** Deneyim ve resmi eğitim, kazancın tartışmasız kralları olmaya devam ediyor. Diğer tüm faktörler sabit tutulduğunda, bir üniversite diploması yıllık ortalama **13.640$** değerinde bir avantaj sağlarken, her bir yıllık mesleki deneyim maaşa yaklaşık **2.498$** ekliyor.
* 🤖 **"Yapay Zeka Maaş Paradoksu":** Verilerin en büyüleyici olduğu nokta burası. Başlangıçta yaptığımız basit bir T-Testi, YZ *kullanmayan* geliştiricilerin daha fazla kazandığını gösterdi. Ancak regresyon analizimiz, bunun kıdemden kaynaklanan istatistiksel bir yanılsama olduğunu kanıtladı; çünkü sektörde zaten çok kazanan kıdemli geliştiriciler, YZ'yi gençlere göre daha yavaş benimsiyordu. Deneyim faktörünü kontrol altına aldığımızda, YZ kullanımının maaş üzerindeki etkisi tamamen anlamsızlaştı (**p = 0,587**). Kısacası: Size daha fazla parayı YZ değil, deneyiminiz kazandırıyor.
* 🏠 **Uzaktan Çalışma Avantajı:** Evden çalışmanın net bir finansal karşılığı var. Tamamen uzaktan (remote) çalışan geliştiriciler, hibrit rollere kıyasla ortalama **11.120$** daha fazla kazanırken, ofise gitme zorunluluğu olan roller belirgin bir finansal kayıp yaşıyor.

📄 **Proje Raporu:** [Stack Overflow Survey Analysis Report (PDF)](StackOverflow_Survey_Analysis_Report.pdf)



## Yazar/Author: Melih Tuğrul Erdoğdu
