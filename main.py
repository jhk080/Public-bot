import datetime
import os
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def send_telegram_message(token, chat_id, message):
  url = f"https://api.telegram.org/bot{token}/sendMessage"
  payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
  try:
    requests.post(url, json=payload)
  except Exception as e:
    print(f"텔레그램 전송 실패: {e}")


def get_naver_issue_keywords():
  """네이버 증권 뉴스에서 주요 이슈 키워드 및 종목명 추출"""
  try:
    url = "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2=258"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    titles = soup.select(".list_body newsflash_body a, .type2 dt a")
    text_corpus = " ".join([t.get_text() for t in titles])
    return text_corpus
  except Exception as e:
    print(f"뉴스 수집 오류: {e}")
    return ""


def main():
  token = os.environ.get("TELEGRAM_TOKEN")
  chat_id = os.environ.get("TELEGRAM_CHAT_ID")

  if not token or not chat_id:
    print("텔레그램 토큰 또는 Chat ID가 설정되지 않았습니다.")
    return

  print("=== [스윙 스크리닝] 데이터 분석 시작 ===")
  df_krx = fdr.StockListing("KRX")

  end_date = datetime.datetime.now().strftime("%Y-%m-%d")
  start_date = (
      datetime.datetime.now() - datetime.timedelta(days=180)
  ).strftime("%Y-%m-%d")

  issue_news_text = get_naver_issue_keywords()
  recommendations = []

  # KOSPI, KOSDAQ 전 종목 대상 스크리닝
  for idx, row in df_krx.iterrows():
    code = row["Code"]
    name = row["Name"]

    try:
      df = fdr.DataReader(code, start_date, end_date)
      if len(df) < 130:
        continue

      # 1. 거래대금 조건: 최근일 거래대금 100억 이상 (종가 * 거래량)
      recent_trading_val = (df["Close"].iloc[-1] * df["Volume"].iloc[-1]) / (
          100000000
      )
      if recent_trading_val < 100:
        continue

      # 2. 일목균형표 지표 계산 ([수식1] 반영)
      high_9 = df["High"].rolling(window=9).max()
      low_9 = df["Low"].rolling(window=9).min()
      df["전환선"] = (high_9 + low_9) / 2

      high_26 = df["High"].rolling(window=26).max()
      low_26 = df["Low"].rolling(window=26).min()
      df["기준선"] = (high_26 + low_26) / 2

      df["선행1"] = (df["전환선"] + df["기준선"]) / 2

      high_52 = df["High"].rolling(window=52).max()
      low_52 = df["Low"].rolling(window=52).min()
      df["선행2"] = (high_52 + low_52) / 2

      df["구름상단"] = np.maximum(df["선행1"], df["선행2"])
      df["구름하단"] = np.minimum(df["선행1"], df["선행2"])

      # 세력선 (112일 이평선)
      df["세력선"] = df["Close"].rolling(window=112).mean()

      # DMI (DI+, DI-) 간이 계산 (14)
      plus_dm = df["High"].diff()
      minus_dm = df["Low"].diff() * -1
      plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
      minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

      tr1 = df["High"] - df["Low"]
      tr2 = abs(df["High"] - df["Close"].shift(1))
      tr3 = abs(df["Low"] - df["Close"].shift(1))
      tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
      atr = tr.rolling(14).mean()

      df["DI_P"] = (pd.Series(plus_dm).rolling(14).mean() / atr) * 100
      df["DI_M"] = (pd.Series(minus_dm).rolling(14).mean() / atr) * 100

      # 3. 수급 조건: 거래량이 전일 대비 3배 이상 AND 양봉
      vol_cond = (
          df["Volume"].iloc[-1] > df["Volume"].iloc[-2] * 3
      ) and (df["Close"].iloc[-1] > df["Open"].iloc[-1])

      c = df["Close"].iloc[-1]
      cloud_bottom = df["구름하단"].iloc[-1]
      cloud_top = df["구름상단"].iloc[-1]
      seoryeok = df["세력선"].iloc[-1]

      # 상향돌파 조건 (구름상단 또는 세력선 돌파)
      cross_cloud = (df["Close"].iloc[-2] <= df["구름상단"].iloc[-2]) and (
          c > cloud_top
      )
      cross_seoryeok = (df["Close"].iloc[-2] <= df["세력선"].iloc[-2]) and (
          c > seoryeok
      )

      # 4. 최종 조건 결합
      cond_final = (
          (c >= cloud_bottom)
          and (cross_cloud or cross_seoryeok)
          and vol_cond
          and (df["DI_P"].iloc[-1] > df["DI_M"].iloc[-1])
      )

      # 네이버 뉴스 이슈 동반 여부 확인
      news_matched = name in issue_news_text

      if cond_final:
        recommendations.append({
            "name": name,
            "code": code,
            "close": c,
            "trading_val": recent_trading_val,
            "news": "🔥 네이버 뉴스 이슈 동반" if news_matched else "📈 기술적 조건 충족",
        })

    except Exception:
      continue

  # 텔레그램 메시지 포맷팅 및 전송
  if recommendations:
    msg = "🚨 *[대세 상승 초입 신호 스윙 추천주]* 🚨\n\n"
    for r in recommendations[:10]:
      msg += f"• *{r['name']}* (`{r['code']}`)\n"
      msg += (
          f"  - 종가: `{r['close']:,}`원 | 거래대금: `{r['trading_val']:.1f}`억원\n"
      )
      msg += f"  - 상태: {r['news']}\n\n"
    msg += "※ 본 정보는 수식 기반 자동 스크리닝 결과로 투자 참고용입니다."
  else:
    msg = (
        "📊 *[오후 3시 스윙 스크리닝 결과]*\n\n오늘 조건을 만족하는 종목이 없습니다."
    )

  send_telegram_message(token, chat_id, msg)
  print("=== 스크리닝 및 전송 완료 ===")


if __name__ == "__main__":
  main()
