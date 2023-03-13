package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)

	modelsDir := "."
	modelName := textencoding.DefaultModel
	m, err := tasks.Load[textencoding.Interface](&tasks.Config{
		ModelsDir: modelsDir, 
		ModelName: modelName,
	})

	if err != nil {
		log.Fatal().Err(err).Send()
	}

	defer tasks.Finalize(m)


	start := time.Now()
	r1, err := m.Encode(context.Background(), "see you later", int(bert.MeanPooling))
	elapsed := time.Since(start)
	if err != nil {
		log.Fatal().Err(err).Send()
	}


	log.Info().Int("elapsed_ms", int(elapsed.Milliseconds())).Send()

	f1 := r1.Vector.Data().F64()

	r2, err := m.Encode(context.Background(), "see you tomorrow", int(bert.MeanPooling))
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	f2 := r2.Vector.Data().F64()

	fmt.Println(Cosine(f1, f2))
}

func Cosine(a []float64, b []float64) (cosine float64, err error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}
