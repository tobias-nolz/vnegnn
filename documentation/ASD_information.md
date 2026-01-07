# Allosteric Database (ASD)

Allosteric Database (ASD) provides a platform for exhaustive information on allosteric proteins and their modulators.
The database now contains 2,422 allosteric proteins from 425 species and 100,320 modulators in three categories (
activators, inhibitors, and regulators). Proteins are annotated with detailed descriptions of allostery, biological
process, and related diseases, as well as modulators with binding affinity, physicochemical properties, and therapeutic
area.

Find more information about the Allosteric Database (ASD) at [ASD Official Website](http://mdl.shsmu.edu.cn/ASD/).

## Overview

The Allosteric Database contains multiple files with different data related to allosteric proteins, modulators, and
sites. Below is a brief description of each file included in the database.

### ASD Release File List

| File Name                                 | File Description                                                                                | File Type     | Version Number | Create Date                    |
|:------------------------------------------|:------------------------------------------------------------------------------------------------|:--------------|:---------------|:-------------------------------|
| **ASD_Release_202309_AS.tar.gz**          | This file contains allosteric site description of all available entries contained in ASD        | Tab delimited | Release 5.1    | 2023-09-20, September 20, 2023 |
| **ASD_Release_202306_2D.tar.gz**          | This file contains 2D structure files of all modulators contained in ASD                        | Mol           | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202306_3D.tar.gz**          | This file contains 3D structure files of all modulators contained in ASD                        | Mol2          | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202306_XF.tar.gz**          | This file contains XML format data of all protein entries contained in ASD                      | XML           | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202306_AP.tar.gz**          | This file contains potential allosteric sites of human proteins in AlloSite-Potential V2 of ASD | Tab delimited | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202306_HL.tar.gz**          | This file contains all Hit-to-Lead records of ASD                                               | Tab delimited | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202309_PP.tar.gz**          | This file contains all targets and modulators for Allo-PPI feature of ASD                       | Tab delimited | Release 5.1    | 2023-09-20, September 20, 2023 |
| **ASD_Release_202306_DU.tar.gz**          | This file contains all dualsteric modulators of ASD                                             | Tab delimited | Release 5.01   | 2023-06-01, June 1, 2023       |
| **ASD_Release_202208_AP_V2.tar.gz**       | This file contains selected potential allosteric sites for DeepAlloDriver                       | Tab delimited | Release 4.22   | 2022-08-16, August 16, 2022    |
| **ASD_Release_202208_AS_Filtered.tar.gz** | This file contains allosteric sites for DeepAlloDriver                                          | Tab delimited | Release 4.22   | 2022-08-16, August 16, 2022    |
| **ASD_Release_201909_AM.tar.gz**          | This file contains all allosteric mutations in Allo-Mutation of ASD                             | Tab delimited | Release 4.10   | 2019-09-26, September 26, 2019 |
| **ASD_Release_201909_DR.tar.gz**          | This file contains all allosteric drugs in Allo-Drug of ASD                                     | Tab delimited | Release 4.10   | 2019-09-26, September 26, 2019 |

### Version

The work in this repository is currently performed only on the `ASD_Release_202309_AS.tar.gz` with version number
`Release 5.1`, created on September 20, 2023.

As by the declaration on their website, the data may not be distributed by a third party and thus, will not be provided
in this repository. Please visit the [ASD Official Website](http://mdl.shsmu.edu.cn/ASD/) to download the data files
directly from the source.

## Data Details

The `ASD_Release_202309_AS.tar.gz` file contains the following information:

| Category                             | Column                    | Description                                                                                     |
|--------------------------------------|---------------------------|-------------------------------------------------------------------------------------------------|
| **Target Information**               | `target_id`               | Unique identifier assigned by ASD.                                                              |
| **Target Information**               | `target_gene`             | The gene symbol.                                                                                |
| **Target Information**               | `organism`                | The species the protein belongs to.                                                             |
| **Target Information**               | `pdb_uniprot`             | UniProt Accession ID                                                                            |
| **Allosteric Structure Information** | `allosteric_pdb`          | PDB ID of the protein structure where the allosteric event occurs.                              |
| **Allosteric Structure Information** | `modulator_serial`        | Serial number of the modulator in the ASD database.                                             |
| **Allosteric Structure Information** | `modulator_alias`         | Alternative name or identifier for the modulator.                                               |
| **Allosteric Structure Information** | `modulator_chain`         | Chain identifier in the PDB structure where the modulator binds.                                |
| **Modulator Information**            | `modulator_class`         | High-level classification of the modulator (e.g., `lig`, `ion`, ...)                            |
| **Modulator Information**            | `modulator_feature`       | Functional effect of the modulator (`activator`, `inhibitor`, `regulator`)                      |
| **Modulator Information**            | `modulator_name`          | IUPAC or common name of the modulator.                                                          |
| **Modulator Information**            | `modulator_resi`          | Residue or PDB "residue" index where the modulator binds.                                       |
| **Functional Annotation**            | `function`                | Describes the functional role of the modulator binding site.                                    |
| **Functional Annotation**            | `position`                | Classifies where the modulator binds (e.g. `Protein-Protein Interaction`, `Inner Protein`, ...) |
| **Publication Information**          | `pubmed_id`               | PubMed ID of the publication describing the allosteric event.                                   |
| **Publication Information**          | `ref_title`               | Title of the referenced publication.                                                            |
| **Allosteric Site Annotation**       | `site_overlap`            | Indicates if the allosteric site overlaps with an active site or other known site.              |
| **Allosteric Site Annotation**       | `allosteric_site_residue` | List of residues that make up the allosteric site.                                              |

On the ASD website, you can find the data for all included proteins (
see [ASD Protein Entries](https://mdl.shsmu.edu.cn/ASD/module/proteins/proteins.jsp)) and modulators (
see [ASD Modulator Entries](https://mdl.shsmu.edu.cn/ASD/module/modulators/modulators.jsp)).

## Citation

[Unraveling allosteric landscapes of allosterome with ASD.](https://academic.oup.com/nar/article/48/D1/D394/5608996?login=false)
Liu X, Lu S, Song K, Shen Q, Ni D, Li Q, He X, Zhang H, Wang Q, Chen Y, Li X, Wu J, Sheng C, Chen G, Liu Y, Lu X, Zhang
J.
Nucleic Acids Research (2020) Database Issue 48: D394–D401.



