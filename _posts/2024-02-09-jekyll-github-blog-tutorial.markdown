---
layout: post
title:  "Jekyll로 GitHub 블로그 만들기 - 설치부터 배포까지"
date:   2024-02-09 02:00:00 +0900
categories: jekyll github-pages tutorial
---

GitHub Pages와 Jekyll을 사용하여 개인 블로그를 만드는 전체 과정을 정리해보았습니다.

## Jekyll과 GitHub Pages를 선택한 이유

1. **무료 호스팅**
   - GitHub Pages를 통해 무료로 웹사이트 호스팅
   - 별도의 서버 비용이 들지 않음

2. **마크다운 지원**
   - 마크다운으로 쉽게 글 작성 가능
   - 코드 하이라이팅 기본 지원 

3. **버전 관리**
   - Git을 통한 모든 컨텐츠의 버전 관리
   - 실수로 삭제해도 복구 가능

4. **커스터마이징**
   - 테마를 통한 쉬운 디자인 변경
   - 필요한 기능을 직접 추가 가능

## 설치 과정

### 1. 준비물
- Ruby 설치 (rbenv 사용)
- Git 설치
- GitHub 계정

### 2. Ruby 환경 설정

```bash
# Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# rbenv 설치
brew install rbenv ruby-build

# rbenv 초기화
rbenv init

# Ruby 설치
rbenv install 3.2.0
rbenv global 3.2.0
```

### 3. Jekyll 설치

```bash
gem install jekyll bundler
```

### 4. 블로그 생성 및 설정

```bash
# 새 Jekyll 사이트 생성
jekyll new USERNAME.github.io
cd USERNAME.github.io

# 의존성 설치
bundle install
```

### 5. GitHub 저장소 설정
1. GitHub에서 새 저장소 생성 (이름: USERNAME.github.io)
2. 로컬 저장소와 연결

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/USERNAME.github.io.git
git push -u origin main
```

## 사용 방법

### 새 글 작성하기
1. `_posts` 폴더에 `YYYY-MM-DD-제목.markdown` 형식으로 파일 생성
2. 파일 상단에 YAML 프론트매터 작성 

```yaml
---
layout: post
title: "글 제목"
date: YYYY-MM-DD HH:MM:SS +0900
categories: category-name
---
```

### 로컬에서 테스트

```bash
bundle exec jekyll serve
```
http://localhost:4000 에서 확인

### 배포

```bash
git add .
git commit -m "Add new post"
git push
```

## 유용한 팁

1. **이미지 추가**
   - `assets` 폴더에 이미지 저장
   - 마크다운에서 `![이미지설명](/assets/이미지파일명)` 형식으로 사용

2. **테마 변경**
   - `_config.yml` 파일에서 theme 설정 변경
   - GitHub Pages에서 지원하는 테마 사용 가능

3. **댓글 기능**
   - Disqus, Utterances 등 추가 가능
   - 정적 사이트의 한계 극복

## 주의사항

1. 파일명은 반드시 날짜-제목 형식을 지켜야 함
2. 프론트매터(YAML)는 필수
3. 빌드 시간은 보통 1-2분 소요

## 참고 자료
- [Jekyll 공식 문서](https://jekyllrb.com/docs/)
- [GitHub Pages 문서](https://docs.github.com/en/pages)