/**
 * @superinstance/triplet-miner
 * Mine (anchor, positive, negative) triplets from git history for contrastive learning
 */

// ── Types ──────────────────────────────────────────────────────────────────

export class Triplet {
  constructor(
    public readonly anchor: string,
    public readonly positive: string,
    public readonly negative: string,
    public readonly similarity: number,
    public readonly source: string,
    public readonly metadata: Record<string, unknown> = {}
  ) {}

  toJSON(): object {
    return {
      anchor: this.anchor,
      positive: this.positive,
      negative: this.negative,
      similarity: this.similarity,
      source: this.source,
      metadata: this.metadata,
    };
  }
}

export enum MiningStrategy {
  RANDOM = "random",
  HARD_NEGATIVE = "hard_negative",
  SEMI_HARD = "semi_hard",
  DOMAIN_AWARE = "domain_aware",
}

export class RoutingDecision {
  constructor(
    public readonly device: string,
    public readonly reason: string,
    public readonly confidence: number
  ) {}
}

// ── Quality Filter ─────────────────────────────────────────────────────────

export class QualityFilter {
  private triplets: Triplet[];

  constructor(triplets: Triplet[]) {
    this.triplets = [...triplets];
  }

  minSimilarity(threshold: number): QualityFilter {
    this.triplets = this.triplets.filter((t) => t.similarity >= threshold);
    return this;
  }

  deduplicate(): QualityFilter {
    const seen = new Set<string>();
    this.triplets = this.triplets.filter((t) => {
      const key = `${t.anchor}||${t.positive}||${t.negative}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
    return this;
  }

  languageFilter(languages: string[]): QualityFilter {
    const exts = new Set(languages.map((l) => `.${l}`));
    this.triplets = this.triplets.filter((t) => {
      const src = t.metadata.file as string | undefined;
      if (!src) return true;
      const ext = src.substring(src.lastIndexOf("."));
      return exts.has(ext) || languages.length === 0;
    });
    return this;
  }

  collect(): Triplet[] {
    return this.triplets;
  }
}

// ── Triplet Miner ──────────────────────────────────────────────────────────

export class TripletMiner {
  private strategy: MiningStrategy;
  private allTriplets: Triplet[] = [];

  constructor(strategy: MiningStrategy = MiningStrategy.RANDOM) {
    this.strategy = strategy;
  }

  async mineFromRepos(repoPaths: string[]): Promise<Triplet[]> {
    for (const repo of repoPaths) {
      await this.mineFromRepo(repo);
    }
    return this.allTriplets;
  }

  async mineFromRepo(repoPath: string): Promise<Triplet[]> {
    // In a real implementation, this would parse git history
    // For now, create representative triplets from repo structure
    const { execSync } = await import("child_process");

    let commits: string[] = [];
    try {
      const log = execSync(`git -C "${repoPath}" log --oneline -50 --format="%H"`, {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      });
      commits = log.trim().split("\n").filter(Boolean);
    } catch {
      // Not a git repo or git unavailable — return empty
      return [];
    }

    if (commits.length < 3) return [];

    for (let i = 0; i < commits.length - 2; i++) {
      const anchor = commits[i];
      const positive = commits[i + 1];
      const negative = commits[Math.floor(Math.random() * commits.length)];

      if (negative === anchor || negative === positive) continue;

      const similarity = this.computeSimilarity(anchor, positive, negative);
      const triplet = new Triplet(anchor, positive, negative, similarity, repoPath, {
        strategy: this.strategy,
        index: i,
      });

      this.allTriplets.push(triplet);
    }

    return this.allTriplets;
  }

  private computeSimilarity(_anchor: string, _positive: string, _negative: string): number {
    // Placeholder similarity — real impl would use embeddings
    return Math.random() * 0.5 + 0.3;
  }

  export(): Triplet[] {
    return [...this.allTriplets];
  }

  toDataset(): { anchors: string[]; positives: string[]; negatives: string[] } {
    return {
      anchors: this.allTriplets.map((t) => t.anchor),
      positives: this.allTriplets.map((t) => t.positive),
      negatives: this.allTriplets.map((t) => t.negative),
    };
  }
}

// ── Exporters ──────────────────────────────────────────────────────────────

export function toJSON(triplets: Triplet[], pretty = true): string {
  return JSON.stringify(triplets.map((t) => t.toJSON()), null, pretty ? 2 : 0);
}

export function toCSV(triplets: Triplet[]): string {
  const header = "anchor,positive,negative,similarity,source";
  const rows = triplets.map(
    (t) => `"${t.anchor}","${t.positive}","${t.negative}",${t.similarity},"${t.source}"`
  );
  return [header, ...rows].join("\n");
}
