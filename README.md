1. make API.json and put the API keys in it like this:
```
{
    "API1",
    "API2",
    "API3"
}
```
2. look into the images folder and the listings and see their place.txt file
3. ***images*** and ***listings*** must look like that in github
4. delete your ***vqa_dataset*** and start from scratch
5. ***setup the venv*** using requirements.txt
6. ***run the gem.py*** to create the vqa_dataset. 

For team member 1 (first third of dataset)
- python gem.py --partition 1 --check-main 0 --limit 50000

For team member 2 (second third of dataset)
- python gem.py --partition 2 --check-main 0 --limit 50000

For team member 3 (final third of dataset)
- python gem.py --partition 3 --check-main 0 --limit 50000

Options

generates for 1000 images by default. Use limits argument for more. reset might not be needed.
- --partition (1, 2, or 3): Which section of the dataset to process
- --check-main (0 or 1): Whether to check for main_image_id (0=no, 1=yes)
- --limit (number): Maximum images to process in a single run
- --reset: Start from beginning of partition (ignores saved progress)
