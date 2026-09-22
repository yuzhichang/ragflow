//  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

package component

import (
	"reflect"
	"testing"

	"ragflow/internal/ingestion/component/schema"
)

// A corpus named by id ("93372.md") leaves title_tks with the id alone; when the document
// declares a title in its header block that title must reach the field, filtered.
func TestDeclaredTitleTokens(t *testing.T) {
	header := "Document 44250\nSource URL: https://www.sportskeeda.com/player/pp-ojha\n" +
		"title: Pragyan Ojha\ndate: 2025-01-01\nPersonal Information\n| Full Name | Pragyan Prayash Ojha |\n"
	cases := []struct {
		name   string
		chunks []schema.ChunkDoc
		want   []string
	}{
		{
			// Only the declared-title lines feed the field ("Source URL", "date", "| Full Name |"
			// are not title lines) and label words are dropped. Short content words are KEPT:
			// the >=5 length rule was replaced by English stopword filtering (2026-09-22), because
			// the length rule also removed real words such as "icc", "cup", "john", "star".
			name:   "declared title is tokenised, labels dropped, short words kept",
			chunks: []schema.ChunkDoc{{Text: header}},
			want:   []string{"pragyan", "ojha"},
		},
		{
			name:   "title/name/fullname are all read, duplicates and wikipedia dropped",
			chunks: []schema.ChunkDoc{{Text: "title: Lesley Manyathela - Wikipedia\nname: Lesley Manyathela\nfullname: Lesley Phuti Manyathela\n"}},
			want:   []string{"lesley", "manyathela", "phuti"},
		},
		{
			name:   "no header block means no title",
			chunks: []schema.ChunkDoc{{Text: "AW: I began teaching at Zama Dance School in January 2008.\n"}},
			want:   nil,
		},
		{
			// Pure numbers are still dropped (the corpus file stem is numeric, so a
			// numeric title token would match every document), but the short words that
			// the old >=5 rule removed are now kept.
			name:   "numeric tokens dropped, short words kept",
			chunks: []schema.ChunkDoc{{Text: "title: 1917 Ok Go\n"}},
			want:   []string{"ok", "go"},
		},
		{
			name:   "empty chunks are skipped safely",
			chunks: []schema.ChunkDoc{{}, {Text: ""}},
			want:   nil,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := declaredTitleTokens(c.chunks)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("declaredTitleTokens() = %v, want %v", got, c.want)
			}
		})
	}
}
