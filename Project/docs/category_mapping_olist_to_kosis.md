# Olist 카테고리 (71) ↔ KOSIS 카테고리 매핑 (정정판)

**작성일**: 2026-05-08 (KOSIS 공식 분류표 확인 후 정정)
**상태**: 사용자 검토 대기 (확정 전)
**용도**: Phase 3 v2 합성 코호트 생성 시 카테고리 정보 부여
**근거**: 통계청 「온라인쇼핑동향조사」 상품군 분류표 (사용자 제공)

## KOSIS 공식 분류 (24개 + 합계, "가전·전자·통신기기"는 합산이라 제외)

| 카테고리 | 조사 범위 |
|---|---|
| 컴퓨터 및 주변기기 | PC, 노트북, 프린터, 스피커, 유형의 소프트웨어 등 |
| 가전·전자 | TV, 냉장고, 세탁기, 디지털카메라 등 |
| 통신기기 | 휴대폰, 휴대폰 주변기기, 유무선 전화기, 무전기 등 |
| 서적 | 각종 도서 (e-Book 제외) |
| 사무·문구 | 사무용품, 문구류, 다이어리, 종이류, 필기구 등 |
| 의복 | 남성복, 여성복, 스포츠웨어, 아동·유아복 등 |
| 신발 | 구두, 운동화, 샌들, 실내화, 아동화 등 |
| 가방 | 핸드백, 가방, 여행용 등 |
| 패션용품 및 액세서리 | 모자, 장갑, 스카프, 시계, 금반지, 액세서리 등 |
| 스포츠·레저용품 | 운동용품, 레저용품, 등산화, 등산배낭 등 |
| 화장품 | 화장품, 향수, 화장관련 소품 등 |
| 아동·유아용품 | 기저귀, 유모차, 그네, 아기침대, 보행기, 카시트, 인형, 완구 등 |
| 음·식료품 | 공산품류(커피, 차, 음료, 생수, 설탕, 식용유, 분유 등), 김치, 장류 등 |
| 농축수산물 | 곡물, 육류, 어류, 채소, 과실, 신선식품류 등 |
| 생활용품 | 주방용품, 침구, 비누, 샴푸, 세제, 화장지, **꽃**, 화분 등 |
| 자동차 및 자동차용품 | 자동차, 오토바이, 튜닝/선팅용품, 내비게이션, 블랙박스, 엔진오일 등 |
| 가구 | 장롱, 화장대, 신발장, 책상, 의자 등 |
| 애완용품 | 사료, 장난감, 장신구 등 |
| 여행 및 교통서비스 | 항공권, 교통티켓, 렌터카, 숙박시설 등 |
| 문화 및 레저서비스 | **영화, 공연 등의 예약서비스 (서비스만!)** |
| e쿠폰서비스 | 상품권 등 |
| 음식서비스 | 온라인 주문 후 조리되어 배달되는 음식 (피자, 치킨 등) |
| 기타서비스 | 인화 등 주문제작, 이사, 청소 등 용역서비스, 렌털서비스 |
| 기타 | 문화상품권, **의료기구**, **골동품**, 종교용품, 성인용품, **음반·비디오·악기** 등 |

**중요한 분류 원칙**:
- 음반·비디오·악기 같은 **물리 매체**는 "기타" (문화·레저서비스 아님!)
- "문화 및 레저서비스"는 **예약 서비스만**
- 인터넷 게임·음악·교육서비스 등 **디지털 콘텐츠는 조사 대상 제외**

---

## 매핑 표 (Olist 71 → KOSIS, 정정판)

| # | Olist (영어) | KOSIS 매핑 | 신뢰도 | 비고 |
|---|---|---|---|---|
| 1 | health_beauty | 화장품 | ⚠️ | health 의약품 부분 손실 (의료기구는 "기타") |
| 2 | computers_accessories | 컴퓨터 및 주변기기 | ✅ | |
| 3 | auto | 자동차 및 자동차용품 | ✅ | |
| 4 | bed_bath_table | 생활용품 | ✅ | "침구" 명시 |
| 5 | furniture_decor | 가구 | ✅ | |
| 6 | sports_leisure | 스포츠·레저용품 | ✅ | |
| 7 | perfumery | 화장품 | ✅ | "향수" 명시 |
| 8 | housewares | 생활용품 | ✅ | |
| 9 | telephony | 통신기기 | ✅ | |
| 10 | watches_gifts | 패션용품 및 액세서리 | ✅ | "시계" 명시 |
| 11 | food_drink | 음·식료품 | ✅ | |
| 12 | baby | 아동·유아용품 | ✅ | |
| 13 | stationery | 사무·문구 | ✅ | |
| 14 | tablets_printing_image | 컴퓨터 및 주변기기 | ✅ | "프린터" 명시 |
| 15 | toys | 아동·유아용품 | ✅ | "인형, 완구" 명시 |
| 16 | fixed_telephony | 통신기기 | ✅ | "유무선 전화기" 명시 |
| 17 | garden_tools | 기타 | ❌ | 정원 도구 — 적합 분류 없음 |
| 18 | fashion_bags_accessories | 가방 | ⚠️ | 가방+액세서리 혼합, 가방 주력 가정 |
| 19 | small_appliances | 가전·전자 | ✅ | |
| 20 | consoles_games | 기타 | ⚠️ | **변경**: 게임기 명시적 분류 없음 → 기타 적합 |
| 21 | audio | 가전·전자 | ✅ | |
| 22 | fashion_shoes | 신발 | ✅ | |
| 23 | cool_stuff | 기타 | ❌ | 정의 모호 |
| 24 | luggage_accessories | 가방 | ✅ | "여행용" 명시 |
| 25 | air_conditioning | 가전·전자 | ✅ | **정정**: TV/냉장고 같은 가전 |
| 26 | construction_tools_construction | 기타 | ❌ | 건축자재 |
| 27 | kitchen_dining_laundry_garden_furniture | 가구 | ✅ | |
| 28 | costruction_tools_garden | 기타 | ❌ | 건축 도구 |
| 29 | fashion_male_clothing | 의복 | ✅ | "남성복" 명시 |
| 30 | pet_shop | 애완용품 | ✅ | |
| 31 | office_furniture | 가구 | ✅ | "책상, 의자" 명시 |
| 32 | market_place | 기타 | ❌ | 정의 모호 |
| 33 | electronics | 가전·전자 | ✅ | |
| 34 | home_appliances | 가전·전자 | ✅ | **정정**: TV/냉장고 등 |
| 35 | party_supplies | 기타 | ❌ | 파티용품 |
| 36 | home_confort | 생활용품 | ⚠️ | 정확한 의미 불명 |
| 37 | costruction_tools_tools | 기타 | ❌ | 건축 도구 |
| 38 | agro_industry_and_commerce | 농축수산물 | ⚠️ | 농산업 |
| 39 | furniture_mattress_and_upholstery | 가구 | ✅ | |
| 40 | books_technical | 서적 | ✅ | |
| 41 | home_construction | 기타 | ❌ | 건축 |
| 42 | musical_instruments | 기타 | ✅ | **정정**: "악기" 기타에 명시 |
| 43 | furniture_living_room | 가구 | ✅ | |
| 44 | construction_tools_lights | 기타 | ❌ | 조명 도구 |
| 45 | industry_commerce_and_business | 기타 | ❌ | 산업·상업 |
| 46 | food | 음·식료품 | ⚠️ | 신선식품이면 농축수산물 가능 |
| 47 | art | 기타 | ⚠️ | **정정**: "골동품" 기타에 명시 |
| 48 | furniture_bedroom | 가구 | ✅ | |
| 49 | books_general_interest | 서적 | ✅ | |
| 50 | construction_tools_safety | 기타 | ❌ | 안전 도구 |
| 51 | fashion_underwear_beach | 의복 | ✅ | "남성복, 여성복" 등 |
| 52 | fashion_sport | 의복 | ✅ | **정정**: "스포츠웨어" 의복에 명시 |
| 53 | signaling_and_security | 기타 | ❌ | 신호·보안 |
| 54 | computers | 컴퓨터 및 주변기기 | ✅ | "PC, 노트북" 명시 |
| 55 | christmas_supplies | 기타 | ❌ | 종교용품 가능성 |
| 56 | fashion_female_clothing | 의복 | ✅ | "여성복" 명시 |
| 57 | home_appliances_2 | 가전·전자 | ✅ | **정정** |
| 58 | books_imported | 서적 | ✅ | |
| 59 | drinks | 음·식료품 | ✅ | "음료" 명시 |
| 60 | cine_photo | 가전·전자 | ✅ | "디지털카메라" 명시 |
| 61 | la_cuisine | 생활용품 | ✅ | "주방용품" 명시 |
| 62 | music | 기타 | ⚠️ | **정정**: 음반은 기타. 디지털 음악이면 조사 제외 |
| 63 | home_comfort_2 | 생활용품 | ⚠️ | |
| 64 | small_appliances_home_oven_and_coffee | 가전·전자 | ✅ | |
| 65 | cds_dvds_musicals | 기타 | ✅ | **정정**: "음반·비디오" 기타에 명시 |
| 66 | dvds_blu_ray | 기타 | ✅ | **정정**: 동일 |
| 67 | flowers | 생활용품 | ✅ | **정정**: "꽃, 화분" 명시! |
| 68 | arts_and_craftmanship | 기타 | ⚠️ | "골동품" 가까움 |
| 69 | diapers_and_hygiene | 아동·유아용품 | ✅ | "기저귀" 명시 |
| 70 | fashion_childrens_clothes | 의복 | ✅ | **정정**: "아동·유아복" 의복에 명시 |
| 71 | security_and_services | 기타서비스 | ⚠️ | 보안 서비스 가정 |

---

## 정정 후 통계

| 신뢰도 | 정정 전 | 정정 후 | 변화 |
|---|---|---|---|
| ✅ 명확 | 41개 (57.7%) | **51개** (71.8%) | +10개 |
| ⚠️ 합리적 | 17개 (23.9%) | **7개** (9.9%) | -10개 |
| ❌ 모호 → 기타 | 13개 (18.3%) | 13개 (18.3%) | 동일 |

→ **명확한 매핑이 71.8%로 증가**, 모호한 매핑은 동일 (KOSIS에 정의 자체 없음)

---

## 사용자 검토 요청 사항

### 1. ⚠️ 남은 7개 (대안 가능)

| Olist | 현재 매핑 | 대안 | 결정 필요 |
|---|---|---|---|
| health_beauty | 화장품 | 일부 의료기구는 "기타" | beauty 위주? 화장품 OK? |
| fashion_bags_accessories | 가방 | 패션용품 및 액세서리 | 가방+액세서리 혼합, 어느 쪽? |
| consoles_games | 기타 | 가전·전자 | 게임기 분류 — 기타 적합? |
| home_confort, home_comfort_2 | 생활용품 | (정의 불명) | OK? |
| agro_industry_and_commerce | 농축수산물 | 기타 | 농업 관련, OK? |
| food | 음·식료품 | 농축수산물 | 가공 vs 신선 — 가공식품 가정? |
| art, arts_and_craftmanship | 기타 | (적합 X) | OK? |
| music | 기타 | 문화·레저서비스 | **단 디지털이면 조사 제외**, 음반이면 기타 |
| fashion_childrens_clothes | 의복 | 아동·유아용품 | 의복으로 정정했는데 OK? |
| security_and_services | 기타서비스 | 기타 | 보안 서비스 가정 OK? |

### 2. ❌ 13개 모호 매핑 (모두 "기타")
- 건축 도구류 5개, garden_tools, cool_stuff, market_place, party_supplies
- home_construction, signaling_and_security, christmas_supplies, industry_commerce_and_business

→ "기타"로 통합 OK? 또는 합성에서 제외?

### 3. KOSIS 카테고리 미사용 (Olist에 해당 없음)
- e쿠폰서비스, 여행 및 교통서비스, 음식서비스
- → Olist는 물리 상품 위주라 자연스러움

---

## 다음 단계

매핑 확정 후:
1. CSV로 저장 (코드용)
2. 합성 코호트 재생성 (cohort_kr_v3.parquet)
3. **카테고리별 m_i 자료 검색** (한국 도소매업/이커머스 영업이익률)
4. env, optim, agents 수정
5. PPO 재학습 + 조건별 분할 평가
