COURSE DATA RELEASE PACKAGE
Hotel Booking Demand — Course Release v1

Purpose
This archive provides a fixed snapshot of the course dataset together with minimal, artefacts that
support data provenance, integrity verification, and methodological governance for a reproducible clustering study.

Contents
1) hotel_bookings_course_release_v1.csv
   - Authoritative dataset snapshot for all experiments and reported results.

2) SHA256SUMS.txt
   - Cryptographic SHA-256 digests for integrity checking and traceability.
     Use these digests to (i) verify that your local copy matches “course release v1” exactly and
     (ii) report the dataset hash in your dataset documentation.

3) DATASET_MANIFEST.yml
   - Machine-readable dataset “data card” specifying: release label/version, unit of analysis, source and
     bibliographic reference, stated license/terms, file name, dimensions (rows/columns), and the SHA-256 hash.

4) column_roles.csv
   - Variable-role annotations to support leakage control and responsible variable inclusion:
     * Outcome and post-event variables MUST NOT be used as clustering inputs (they may be retained only for
       post-hoc profiling/interpretation).
     * ID-like or extremely high-cardinality identifiers should be excluded unless a clear justification and a
       responsible encoding strategy are provided.

5) (Optional) subsample_indices_v1_n30000_seed12345.txt
   - Reproducible subsample specification (0-based row indices) intended for computationally demanding methods.
     If used, report the sampling rule (sample size + seed/indices) and discuss its impact on results.

Repository policy
Do NOT commit the raw CSV dataset to version control. Instead, include clear instructions and/or scripts to obtain
the dataset and reproduce the complete preparation pipeline from the released snapshot.



