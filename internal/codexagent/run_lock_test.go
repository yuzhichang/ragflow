//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package codexagent

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestKeyedLocksSerializes is the guarantee the session lock exists for: two turns for the
// same (tenant, session) never run at the same time.
func TestKeyedLocksSerializes(t *testing.T) {
	k := newKeyedLocks()
	key := sessionKey("t", "s")

	var (
		inCritical int32
		overlapped int32
		wg         sync.WaitGroup
	)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l := k.acquire(key)
			if atomic.AddInt32(&inCritical, 1) != 1 {
				atomic.StoreInt32(&overlapped, 1)
			}
			time.Sleep(2 * time.Millisecond)
			atomic.AddInt32(&inCritical, -1)
			k.release(key, l)
		}()
	}
	wg.Wait()
	if overlapped == 1 {
		t.Fatal("same-session critical sections overlapped")
	}
}

// TestKeyedLocksDifferentKeysDoNotContend guards the keying: a different session must not
// wait on this one (a wrong key would either deadlock here or over-serialize).
func TestKeyedLocksDifferentKeysDoNotContend(t *testing.T) {
	k := newKeyedLocks()
	a := k.acquire(sessionKey("t", "a"))
	defer k.release(sessionKey("t", "a"), a)
	b := k.acquire(sessionKey("t", "b")) // must return immediately
	k.release(sessionKey("t", "b"), b)
}

// TestKeyedLocksForgetPrunes is the leak fix: forget prunes an idle entry immediately, and
// a held entry only once its holder releases — never orphaning an in-flight turn.
func TestKeyedLocksForgetPrunes(t *testing.T) {
	k := newKeyedLocks()

	// Held: forget marks; the entry must survive until release.
	key := sessionKey("t", "held")
	l := k.acquire(key)
	k.forget(key)
	k.mu.Lock()
	_, present := k.locks[key]
	k.mu.Unlock()
	if !present {
		t.Fatal("forget must not drop a lock that is still held")
	}
	k.release(key, l) // refs -> 0 and forgotten => pruned
	k.mu.Lock()
	n := len(k.locks)
	k.mu.Unlock()
	if n != 0 {
		t.Fatalf("entry not pruned after release: %d remain", n)
	}

	// Idle: forget prunes at once.
	idle := sessionKey("t", "idle")
	k.release(idle, k.acquire(idle))
	k.forget(idle)
	k.mu.Lock()
	n = len(k.locks)
	k.mu.Unlock()
	if n != 0 {
		t.Fatalf("idle forget did not prune: %d remain", n)
	}
}
