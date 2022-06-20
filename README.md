# Mini Project
PUBG-Finish-Place-Prediction

## 프로젝트 설명
- 머신러닝 워크플로우를 완벽하게 이해하고 사용하기 위한 프로젝트 입니다.
- 조별 프로젝트 입니다.
- Kaggle의 데이터를 사용합니다.(https://www.kaggle.com/c/pubg-finish-placement-prediction)

## 조원 및 실험 결과 공유 Notion
- 조원 : 노치현, 이민석, 조한길
- Notion Page : https://ahead-canidae-162.notion.site/2-29-e0dc512444074302890674ddd9801430
- 발표 ppt : https://docs.google.com/presentation/d/1pCJv883LN13Wo0G6Qk61SEQtrUnr3x7yBE6RgMenVX8/edit?usp=sharing

## 실험 관련 gdrive 링크
 https://drive.google.com/drive/folders/1T9Jn8EgUep7LRxACPjrB4t4fbT1fMCIs?usp=sharing

## 데이터 설명
In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.

You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 player per group.

You must create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).
- PUBG 게임에서는 각 매치(matchId)에 최대 100명의 플레이어가 시작합니다.
- 플레이어는 탈락한 다른 팀이 얼마나 생존해 있는지를 기준으로 게임 종료 시 순위(winPlacePerc)가 매겨지는 팀(groupId)에 속할 수 있습니다.
- 게임에서 플레이어는 다양한 무기를 집어들고, 죽지 않은(knocked) 동료들을 되살리고, 차량을 운전하고, 수영하고, 뛰고, 쏘고, 모든 결과를 경험할 수 있습니다.
- 예를 들어 너무 멀리 떨어지거나, 멀리서 달려오다가(자기장 밖에서) 스스로 죽는 것입니다.
- 각 행에 한 플레이어의 post-game 통계를 포함하도록 포맷된 익명화된 PUBG 게임 통계를 다수 제공합니다.
- 데이터는 모든 타입(솔로, 듀오, 스쿼드, 커스텀 등)의 매치에서 가져옵니다.
- 경기당 100명 또는 그룹당 최대 4명의 플레이어가 있다는 보장은 없습니다.
- 1(1위)부터 0(1위)까지의 scaling 된, 최종 통계를 기준으로 선수의 결승 배치를 예측하는 모델을 만들어야 합니다.

## 변수 설명
- Id - Player’s Id(플레이어 식별 ID)
- groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.(매치 내에서 그룹을 식별하는 정수 ID. 같은 그룹의 플레이어가 다른 경기에서 플레이하는 경우, 매번 다른 그룹 ID를 가집니다.)
- matchId - ID to identify match. There are no matches that are in both the training and testing set.(매치를 식별할 정수 ID. 훈련 세트와 테스트 세트 모두 일치하는 항목이 없습니다.)
- assists - Number of enemy players this player damaged that were killed by teammates.(이 플레이어가 피해를 입힌 적 플레이어 중 동료에게 살해된 플레이어 수.)
- boosts - Number of boost items used.(사용된 부스트(에너지 드링크, 진통제 등) 항목 수.)
- damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.(총 데미지 량. 참고: 자해 제외.)
- DBNOs - Number of enemy players knocked.(넉다운 시킨 플레이어 수.)
- headshotKills - Number of enemy players killed with headshots.(헤드샷으로 제거한 적 플레이어의 수.)
- heals - Number of healing items used.(사용된 치료키트(붕대, 구급상자 등) 항목 수.)
- killPlace - Ranking in match of number of enemy players killed.(죽인 적 플레이어의 수에 따른 매치 내 순위.)
- killPoints -  Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.(킬 기반 플레이어의 외부 순위. (킬만 생각하는 Elo 순위라고 생각하시면 됩니다.))
- kills - Number of enemy players killed.(제거한 적 플레이어의 수.)
- killStreaks - Max number of enemy players killed in a short amount of time.(짧은 시간 내에 제거한 적 플레이어 수 중 최대치.)
- longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.(처치(kill) 시 제거한 플레이어와 플레이어의 가장 긴 거리. - 플레이어를 쓰러뜨리고 차를 몰고 도망가면 가장 긴 Kill stat가 발생할 수 있기 때문에 오해의 소지가 있을 수 있습니다.)
- maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.(매치에서 가장 낮은 순위. 데이터가 순위를 건너뛰기 때문에 이 값은 - - numGroups와 일치하지 않을 수 있습니다.)
- numGroups - Number of groups we have data for in the match.(우리가 가지고 있는 매치 데이터 내 플레이어 그룹 수.)
- revives - Number of times this player revived teammates.(이 플레이어가 팀원을 부활시킨 횟수.)
- rideDistance - Total distance traveled in vehicles measured in meters.(미터 단위로 측정한 차량의 총 주행 거리.)
- roadKills - Number of kills while in a vehicle.(로드킬 횟수.)
- swimDistance - Total distance traveled by swimming measured in meters.(미터 단위로 측정한 수영으로 이동한 총 거리.)
- teamKills - Number of times this player killed a teammate.(팀킬 횟수.)
- vehicleDestroys - Number of vehicles destroyed.(파괴된 차량 수.)
- walkDistance - Total distance traveled on foot measured in meters.(미터 단위로 측정한 도보로 이동한 총 거리.)
- weaponsAcquired - Number of weapons picked up.(주운 무기의 수.)
- winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.(승자 기준 외부 순위. (승자만이 중요한 Elo 순위라고 생각하시면 됩니다.))
- winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.(예측 대상(target). 백분위수 승리 배치로, 1은 1위, 0은 경기 꼴찌에 해당합니다. 이 값은 numGroups가 아닌 maxPlace에서 계산되므로, 매체에 누락된 chunks(상당한 양)가 있을 수 있습니다.)

## 제한 사항
- killPlace 변수의 경우 data leakage가 있습니다.
- https://www.kaggle.com/competitions/pubg-finish-placement-prediction/discussion/79161
- 따라서 해당 변수를 제외하고 프로젝트를 진행하기로 결정했습니다.