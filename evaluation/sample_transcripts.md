# Sample evaluation transcripts

These examples are expected behavior with the included fictional corpus. Similarity thresholds can vary slightly by scikit-learn version.

## Supported query 1
**Question:** What is the refund period?
**Expected answer includes:** "Eligible physical products may be returned within 30 calendar days after delivery."
**Expected citation:** `product_policy.md`, passage containing Returns and refunds.
**Expected support status:** supported.

## Supported query 2
**Question:** How many paid annual leave days do full-time employees receive?
**Expected answer includes:** "Full-time employees receive 18 days of paid annual leave per calendar year."
**Expected citation:** `employee_handbook.txt`.
**Expected support status:** supported.

## Supported query 3
**Question:** How long does a refund take after approval?
**Expected answer includes:** "5–7 business days after approval."
**Expected citation:** `product_policy.md`.
**Expected support status:** supported.

## Unsupported query
**Question:** Who won the 2032 World Cup?
**Expected answer:** `Information not found in provided document.`
**Expected support status:** unsupported.

## Run the transcript tests
Run `python -m unittest discover -s tests -v`.
