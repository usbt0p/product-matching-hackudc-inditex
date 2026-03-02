# Complete the look (InditexTech)

Develop a solution that, starting from an image of a model, identifies the items they are wearing (e.g., dress, heels, necklace, or handbag) and, for each one, returns the corresponding product reference from a predefined catalog.

To assist in the development, the following resources will be provided:

- A set of images of the model wearing the products will be provided; this dataset can be used to develop, train, and validate your models.
- A catalog of available products (including images, descriptions, and unique identifiers) will also be supplied.
- Additionally, a separate set of images of the model without the associated products will be delivered. This set will be used to measure the effectiveness of your solution.
- The task consists of recognizing and linking only the visible items that are included within the provided catalog.
- You will have access to a platform where you can download the datasets, upload your solutions, and view real-time metrics (accuracy and ranking) to compare your performance with other teams.

## Criteria

Scoring criteria is the highest performance score on the separate test set without the associated products, calculated as the percentage of correct lines in the submission over the ground truth.
A maximum of 15 products associated with each bundle will be evaluated (the first 15 rows found for each bundle).

The final `.csv` must contain a header and a pair of `bundle_asset_id,product_asset_id` for each product associated with each bundle.

## Prize 
> Compact AI Computer with Hailo-8 AI Accelerator