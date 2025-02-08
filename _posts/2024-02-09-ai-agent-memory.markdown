---
layout: post
title: "AI Agent의 메모리 관리와 상태 유지"
date: 2024-02-09 15:00:00 +0900
categories: [agent-dev]
description: "AI Agent의 메모리 시스템 구현과 상태 관리 전략을 다룹니다."
---

AI Agent의 메모리 시스템은 지능적인 의사결정과 일관된 대화를 가능하게 하는 핵심 요소입니다. 이 글에서는 AI Agent의 메모리 관리 방법과 다양한 구현 전략을 살펴보겠습니다.

## 메모리 유형

### 1. 단기 메모리 (Short-term Memory)
즉각적인 컨텍스트 유지를 위한 메모리 시스템:

#### 대화 컨텍스트
```python
from langchain.memory import ConversationBufferMemory

# 기본 대화 메모리
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 대화 기록 추가
memory.save_context(
    {"input": "안녕하세요"},
    {"output": "안녕하세요! 무엇을 도와드릴까요?"}
)
```

#### 임시 상태 관리
```python
class WorkingMemory:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.items = []
    
    def add_item(self, item):
        if len(self.items) >= self.capacity:
            self.items.pop(0)  # FIFO
        self.items.append(item)
    
    def get_recent_items(self):
        return self.items
```

### 2. 장기 메모리 (Long-term Memory)
지속적인 지식 저장을 위한 시스템:

#### 벡터 데이터베이스 활용
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 벡터 저장소 초기화
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings
)

# 정보 저장
vectorstore.add_texts(
    texts=["중요한 정보 1", "중요한 정보 2"],
    metadatas=[{"source": "user"}, {"source": "system"}]
)

# 유사 정보 검색
results = vectorstore.similarity_search(
    query="관련 정보를 찾아주세요",
    k=3
)
```

#### 지식 그래프 구현
```python
from networkx import DiGraph
import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = DiGraph()
    
    def add_knowledge(self, subject, predicate, object):
        self.graph.add_edge(subject, object, relation=predicate)
    
    def query_knowledge(self, subject):
        return [
            (n, self.graph[subject][n]['relation'])
            for n in self.graph[subject]
        ]
```

## 메모리 관리 전략

### 1. 컨텍스트 윈도우 관리
```python
class ContextWindow:
    def __init__(self, max_tokens=2000):
        self.max_tokens = max_tokens
        self.messages = []
        self.current_tokens = 0
    
    def add_message(self, message, token_count):
        while self.current_tokens + token_count > self.max_tokens:
            removed = self.messages.pop(0)
            self.current_tokens -= removed['tokens']
        
        self.messages.append({
            'content': message,
            'tokens': token_count
        })
        self.current_tokens += token_count
```

### 2. 메모리 압축
```python
from langchain.memory import ConversationSummaryMemory

# 요약 기반 메모리
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(),
    max_token_limit=1000
)

# 대화 요약 생성
summary = summary_memory.predict_new_summary(
    messages=[],
    existing_summary="이전 대화 요약"
)
```

### 3. 우선순위 기반 저장
```python
class PriorityMemory:
    def __init__(self):
        self.high_priority = []
        self.medium_priority = []
        self.low_priority = []
    
    def store(self, item, priority):
        if priority == "high":
            self.high_priority.append(item)
        elif priority == "medium":
            self.medium_priority.append(item)
        else:
            self.low_priority.append(item)
    
    def retrieve(self, priority_level=None):
        if priority_level == "high":
            return self.high_priority
        elif priority_level == "medium":
            return self.medium_priority
        elif priority_level == "low":
            return self.low_priority
        return (
            self.high_priority +
            self.medium_priority +
            self.low_priority
        )
```

## 고급 메모리 기능

### 1. 감정 상태 추적
```python
class EmotionalMemory:
    def __init__(self):
        self.emotional_state = {
            'valence': 0.0,  # 긍정/부정
            'arousal': 0.0,  # 활성화 정도
            'dominance': 0.0  # 지배력
        }
    
    def update_emotion(self, text):
        # 감정 분석 수행
        analysis = self._analyze_emotion(text)
        
        # 상태 업데이트
        self.emotional_state = {
            k: (v + analysis[k]) / 2
            for k, v in self.emotional_state.items()
        }
```

### 2. 학습된 패턴 저장
```python
class PatternMemory:
    def __init__(self):
        self.patterns = {}
        self.frequency = {}
    
    def observe_pattern(self, sequence):
        pattern = tuple(sequence)
        self.patterns[pattern] = self.patterns.get(pattern, 0) + 1
    
    def get_likely_next(self, current_sequence):
        matching_patterns = [
            p for p in self.patterns
            if p[:-1] == tuple(current_sequence)
        ]
        return max(matching_patterns,
                  key=lambda p: self.patterns[p],
                  default=None)
```

## 메모리 최적화

### 1. 캐싱 전략
```python
from functools import lru_cache

class CachedMemory:
    def __init__(self):
        self.cache = {}
    
    @lru_cache(maxsize=1000)
    def retrieve_expensive_data(self, key):
        # 비용이 큰 데이터 검색 작업
        return self.cache.get(key)
```

### 2. 분산 저장
```python
from redis import Redis

class DistributedMemory:
    def __init__(self):
        self.redis = Redis(host='localhost', port=6379)
    
    def store(self, key, value, expiry=None):
        self.redis.set(key, value, ex=expiry)
    
    def retrieve(self, key):
        return self.redis.get(key)
```

## 보안 고려사항

### 1. 데이터 암호화
```python
from cryptography.fernet import Fernet

class SecureMemory:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def store_secure(self, data):
        encrypted = self.cipher_suite.encrypt(
            str(data).encode()
        )
        return encrypted
    
    def retrieve_secure(self, encrypted_data):
        decrypted = self.cipher_suite.decrypt(encrypted_data)
        return decrypted.decode()
```

## 결론

AI Agent의 메모리 시스템은 지능적인 행동과 일관된 상호작용을 위한 핵심 요소입니다. 적절한 메모리 관리 전략의 선택과 구현은 AI Agent의 성능과 사용자 경험을 크게 향상시킬 수 있습니다.

## 참고 자료
- [메모리 시스템 설계 가이드](https://example.com)
- [최적화 전략](https://example.com)
- [보안 베스트 프랙티스](https://example.com)

## 메모리 시스템 구조

```mermaid
graph TB
    subgraph "메모리 계층"
        A[단기 메모리] --> B[작업 메모리]
        C[장기 메모리] --> B
        B --> D[출력]
    end
    
    subgraph "저장소 유형"
        E[벡터 저장소] --> C
        F[관계형 DB] --> C
        G[그래프 DB] --> C
    end
    
    subgraph "캐시 시스템"
        H[L1 캐시] --> I[L2 캐시]
        I --> J[메인 메모리]
    end
```

## 메모리 처리 흐름

```mermaid
sequenceDiagram
    participant I as 입력
    participant S as 단기메모리
    participant W as 작업메모리
    participant L as 장기메모리
    
    I->>S: 새로운 정보
    S->>W: 컨텍스트 통합
    W->>L: 중요 정보 저장
    L-->>W: 관련 정보 검색
    W->>S: 상태 업데이트
```

## 메모리 컴포넌트

```mermaid
classDiagram
    class MemorySystem {
        +short_term: ShortTermMemory
        +working: WorkingMemory
        +long_term: LongTermMemory
        +store()
        +retrieve()
    }
    class ShortTermMemory {
        +capacity: int
        +retention_time: float
        +items: List
    }
    class WorkingMemory {
        +active_items: Dict
        +context: Context
        +process()
    }
    class LongTermMemory {
        +storage: Storage
        +index: Index
        +query()
    }
    
    MemorySystem *-- ShortTermMemory
    MemorySystem *-- WorkingMemory
    MemorySystem *-- LongTermMemory
``` 