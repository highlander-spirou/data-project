<h1>Phân tích volume trong định lượng rủi ro</h1>

- ngta thường dùng **Volume** và **Log Volume** để thể hiện mối interested của nhà đầu tư đối với 1 asset
- Tuy nhiên, nếu 1 asset bị điều chỉnh, ta sẽ thấy khối lượng giao dịch cũng tăng
- do đó, ngta còn có chỉ số OBV thể hiện trend của giao dịch, which decreases on decrement in price

<br>
<h1>Phân tích đường giá, đường trung bình giá và các phép biến đổi về giá</h1>

- Dùng CandleStick và MAV để vẽ và phân tích sơ bộ đường giá và các điểm đặc biệt trong data

**<h3>Log Returns vs Standardize Returns</h3>**
- Standardize returns có tác dụng biến mọi distribution về mean = 0 và std bằng 1. Tuy nhiên, tác dụng của các biến số > 3th std xô lệch distribution rất là lớn (very very long tailed kurtosis). 
- Log return có tác dụng làm giảm ảnh hưởng của outliners. Biên độ của outliners sẽ đc kéo sát về với mean hơn, khiến distribution ít lepto-kurtosis hơn. Lưu ý: Log-return ko nhận số âm !!!
- Scaler return: Scaler return có tác dụng biến mọi range về -1 -> 1, do đó dễ so sánh mean và std hơn (khó so sánh distribution)

<br>
<h1>Phân tích phân phối trong định lượng rủi ro</h1>

**<h3>Daily Percentage Change (DPC)</h3>**
- Daily percentage change thể hiện sự votality của thị trường. Nó thể hiện sự chênh lệch giá khi mua ngày hôm trước và bán tại ngày hôm sau 
    - ta có: 

            p1 - giá khi mua tại ngày n
            p2 - giá khi bán tại ngày n+1

    - như vậy, công thức DPC đc tính là
    $$ DPC = {p2 \over p1}-1 $$
    - nếu DPC > 1 => giá trị của cổ phiếu có xu hướng tăng
    - nếu DPC < 1 => giá trị của cổ phiếu có xu hướng giảm

    - tập hợp DPC trong 1 khoảng thời gian, ta có 1 histogram. nếu Histogram này càng mập chứng tỏ cổ phiếu này lên nhiều mà xuống cũng nhiều => high votality

    - dựa theo qui luật phân phối, sự mập của 1 histogram do standard deviation & kurtosis của phân phối quyết định. Standard deviation của 1 phân phối DPC đc gọi là `volitality` trong nhiều analysis
    
    - Đối với 2 stocks có standard deviation gần giống nhau, thằng nào càng có kurtosis cao khi thua thì thua càng đậm (tương tự với thắng). Do đó, muốn an toàn khi chơi chứng khoán, chọn thằng stock nào mập mập tí

    - tại 1 khung thời gian xác định, skewness của 2 stock thể hiện tendency của 2 thằng đó tăng hay giảm. do đó, ta có thể tính skewness để so sánh giá trị của từng thằng stock 

    - ***positive skew is good***. Because the mode of the positive skew model is less than the mean - meaning that majority of the time the price of the stock is going down - there is some probability of gaining huge profits (mean) that can cover all the frequent small losses (mode). Skewness is use alot in day-trading to predic the effectieness of high risk - high reward strategy. 

    - Lưu ý, bậc của distribution đc tính từ standard normal distribution (mean=0 và std=1) (tính riêng mỗi thằng thì ăn lz)

**<h3>Cumulative result</h3>**
- Ngoài các con số, box-plot & ECDF của DPC thể hiện rất tốt votality của 1 stock

- Khác với annualized-returns vì chia cho năm -> những năm ăn nên làm ra sẽ lấp liếm cho những năm thua lỗ liên hồi -> ETF và các quỹ đầu tư hay sử dụng chỉ số này. Chỉ số này cực tốt nếu emphasize gains

**<h3>Value at Risk (VaR) và Conditional Value at Risk (CVaR)</h3>**
-  Khi 1 phân phối của *Daily Returns* hình thành trong 1 given timeframe, ta có thể tính toán đc **VaR** và **CVaR (hay còn gọi là expected-shortfall)**. 
- **VaR** đc định nghĩa là the minimum loss in 1 period of time within an alpha confidence (tiếng việt là giá trị `x` của phân phối tại left-alpha)
<br>

- **CVaR, (expected shortfall)** đc định nghĩa là the extreme loss (if a loss may occurs) within an alpha confidence (tiếng việt là giá trị `x` extreme mà 1 phân phối có thể đạt đc left-alpha). Do phân phối chuẩn tiến tới vô cùng, there are no limits to max. do đó, CVaR sẽ quantifies con số tương tự max, định  nghĩa thành extreme loss
- khi **VaR** và **CVaR** càng cao,độ rủi ro kh mất tiền càng lớn.


<h1>Các chỉ số trong định lượng rủi ro</h1>

- **Sharpe ratio**: Sharpe ratio dùng để thể hiện risk-returns ratio, tức "độ worth" của 1 assets so với risk mà nó mang lại. Nếu sharpe-ratio dương tức là tốt (khoảng margin thường từ 0.2-0.3 nếu đầu tư dài hạn, >=0.5 nếu đầu tư trung hạn, >1 tức khoảng đầu tư best vãi cả lz tuy nhiên khó duy trì trong 1 khoảng thời gian dài). Nếu sharpe-ratio <0, tức tài sản có rủi ro *not worth* bằng tài sản ko rủi ro
- **Sortino ratio**: **Sortino ratio** đc sử dụng đặc biệt cho phân tích portfolio (phân bổ đầu tư), dùng để **Sharpe Ratio** with *only negative returns*. ***A higher Sortino ratio is better than a lower one*** as it indicates that the portfolio is operating efficiently by not taking on unnecessary risk. giống như Sharpe ratio, càng lớn càng tốt

- **Information ratio**: Là Sharpe ratio mà thay thế risk-free ratio bằng 1 benchmark

- **M2 ratio**: 
    - Với Sharpe Ratio, ta chỉ biết how much "*worth*" an asset have within a given risk (compare to non-risk return). Sharpe ratio will not tell how much profit you expected to make from that risk.
    - Nó là con lai của Sharpe ratio và Information ratio
    - M2 measure helps in knowing that with the given amount of risk taken, how much the portfolio will reward an investor, bằng cách nhân Sharpe ratio với standard deviation
    - Trong phân tích rủi ro, kết quả của **M2-Ratio** nhân với initial-cost => ***risk-adjusted returns***
    - Vì measure trực tiếp với lợi nhuận, thằng nào có **M2-Ratio** cao hơn thằng đó làm cha
- **Maximum Drawdown (MDD)**: khi 1 đỉnh rớt xuống 1 đáy, nó sẽ tạo ra 1 drawdown. Maximum drawdown là khoảng cách lớn nhất của 1 drawdown. Maximum drawdown sẽ đc tính theo 1 khung thời gian, đc so sánh giữa các assets để so sánh độ rủi ro. ****LƯU Ý***: MDD so sánh magnitude of falling, taken out of the context of the peak. vd: 1 thằng chỉ loay hoay tại 4 -> 1 sẽ ko thể nào đáng so sánh bằng thằng từ 100 -> 60 đc, do đó, nó phải đc kết hợp với các chỉ số khác
- **Calmar ratio**: là 1 chỉ số tương tự **M2-ratio**, nó measures the return per unit of risk. ****LƯU Ý***: **Calmar Ratio** measures annually, not by month or day
- **Chỉ số Beta**: chỉ số beta dùng để measure độ volatility so với S&P500 (với S&P500 =1, nếu Beta = 1.5, meaning that nếu S&P tăng hoặc giảm 1, asset sẽ tăng (hoặc giảm) 1.5). Have beta values of less than 1, reflecting their lower volatility as compared to the broad market. Have beta values of more than 1. These types of securities have greater volatility.

<br> 

<h1>Chỉ số lạm phát</h1>

- Chỉ số lạm phát đc tính theo cpi là pct_change của CPI nhân với 100
- Khi ta vẽ 2 linear regression function của DPC và tỉ lệ lạm phát (nhớ rằng percentage change là 1 normalize function vì có root = 0), ta sẽ lấy đc slope (a trong y=ax+b). Thông thường, ta muốn đường slope của 1 asset tăng mạnh bạo hơn slope của tỉ lệ lạm phát
- Chỉ số lạm phát thường được giữ ở 1 khoảng an toàn (3-4%). Khi chỉ số lạm phát tăng, nó có thể ảnh hưởng tới returns của các assets. Ta có thể isolate các ngày chỉ số lạm phát vượt ngưỡng, và lấy nó để so sánh với giá của các assets
- Chỉ số inflation-adjusted return đc tính bằng công thức:
$$ {Inflation adjusted return = {(1 + Return) \over (1 + Inflation)} - 1} $$
- Cuối cùng, ta vẽ box-plot cho inflation-adjusted return vs percentile của inflation, chỉ số này dùng để so sánh mức sinh lời của các tài sản so với nhau (và so với inflation)


<h1>Danh mục đầu tư và portfolio optimization</h1>

- Danh mục đầu tư (Portfolio) đc tạo nên bằng cách gán weights lên các assets đầu tư
- Khi ta seeding các danh mục đầu tư, ta có thể vẽ đc scatter plot về risk-per-return (risk trục x, return trục y). Đôi khi, ngta assign sharpe-ratio hay Sortino ratio as color-gradient để dễ nhìn hơn
- Theo qui luật xác suất, seeding 1 tập data đủ lớn sẽ tạo ra 1 Brownian-motion, which has a parabollar shape

<img src="./efficient.png" alt="Brownian Motion of Portfolio seeding" style="height: 300px; width:500px;"/>

- Đường màu đỏ biểu thị Efficient Frontier - giá trị tối ưu hóa nhất cho return-per-risk (interpret tương tự area under the curve). Ta có thể lựa chọn điểm mà danh mục đầu tư cho lại kết quả desire nhất từ Efficient Frontier.


<h1>Đánh chứng khoán suggestion</h1>

- **On-Balance Volume**: On-Balance Volume là chỉ số thường dùng để phân tích trend của từng loại assets, nó ko mang tính so sánh giữa các assets với nhau. Khi trend của 1 asset tăng, đường OBV thường tăng nhanh hơn đường chạy giá, ngược lại, khi giảm thì OBV giảm nhanh hơn dường chạy giá


- **chỉ số RSI**: chỉ số RSI thường dùng để đánh chứng khoán thay vì quản lí risk. chỉ số RSI sẽ tạo 1 vùng 70-30 để dự đoán điểm đảo chiều của 1 tài sản.



<br>

- Giải quyết stock-price prediction bằng machine learning:
    - Linear regression
    - Multiple
    - Multivariate Adaptive Regression Splines (MARS)
    - Hidden Markov Chain
    - Decision Tree
    - Bagging regression
    - Gradient Boosting regression
    - XGBoost
    - Facebook's prophet ???


